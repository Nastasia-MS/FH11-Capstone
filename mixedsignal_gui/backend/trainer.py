from PySide6.QtCore import QThread, Signal
import numpy as np
import os
import time
import json
import concurrent.futures

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

from .torch_models import get_model

# Models that expect 2-channel IQ input
IQ_MODELS = {'ResNet1DOptimized'}


class TrainerThread(QThread):
    # Signals: epoch, total_epochs, train_loss, val_loss, train_acc, val_acc
    progress = Signal(int, int, float, float, float, float)
    finished = Signal(str)

    # Match the notebook's signal length for ResNet1D
    TARGET_LENGTH = 2048

    def __init__(self, file_label_pairs, labels, model_name='SimpleCNN', epochs=10,
                 batch_size=32, lr=0.001, val_split=0.2,
                 weight_decay=1e-4, label_smoothing=0.1, grad_clip=1.0,
                 model_hparams=None, save_dir=None, device='cpu'):
        super().__init__()
        self.file_label_pairs = list(file_label_pairs)
        self.labels = list(labels)
        self.model_name = model_name
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_split = float(val_split)
        self.weight_decay = float(weight_decay)
        self.label_smoothing = float(label_smoothing)
        self.grad_clip = float(grad_clip)
        self.model_hparams = model_hparams or {}
        self.save_dir = save_dir
        
        # Use provided device parameter instead of auto-detecting
        self.device = torch.device(device)
        self._stop = False
        
        if self.device.type == 'cuda':
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                print(f"Using device: {self.device} - {torch.cuda.get_device_name(0)}")
            else:
                print(f"Warning: CUDA device requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
                print(f"Using device: {self.device}")
        else:
            print(f"Using device: {self.device}")

    def stop(self):
        """Request training to stop after the current epoch."""
        self._stop = True

    def _load_array(self, path):
        try:
            if path.lower().endswith('.npy') or path.lower().endswith('.npz'):
                arr = np.load(path, allow_pickle=True)
                # npz -> first array
                if isinstance(arr, np.lib.npyio.NpzFile):
                    keys = list(arr.keys())
                    arr = arr[keys[0]] if keys else None
                return np.asarray(arr, dtype=np.float32)
            if path.lower().endswith('.csv'):
                return np.loadtxt(path, delimiter=',').astype(np.float32)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
        return None

    @property
    def _is_iq_model(self):
        """Return True if the selected model expects 2-channel IQ input."""
        return self.model_name in IQ_MODELS

    def _prepare_iq_data(self, X_flat):
        """Split a complex/interleaved signal into 2-channel (I, Q) format.

        Each row in X_flat is a 1D signal.  The signal may be:
        - Already complex-valued  -> split real / imag
        - Real-valued but interleaved I/Q -> split even / odd indices
        Returns shape (N, 2, L) where L is half the original length.
        """
        if np.iscomplexobj(X_flat):
            I = X_flat.real
            Q = X_flat.imag
            return np.stack([I, Q], axis=1)

        # Real-valued: treat as interleaved I/Q
        # Make sure length is even
        L = X_flat.shape[1]
        if L % 2 != 0:
            X_flat = X_flat[:, :L - 1]
            L = L - 1
        half = L // 2
        I = X_flat[:, 0::2]  # even indices
        Q = X_flat[:, 1::2]  # odd indices
        return np.stack([I, Q], axis=1)

    def _prepare_1ch_data(self, X_flat):
        """Reshape flat data to (N, 1, L) for 1-channel Conv1d models."""
        return X_flat[:, np.newaxis, :]

    def _normalize_iq(self, X_iq):
        """Per-sample power normalization for IQ data (N, 2, L)."""
        power = np.mean(X_iq[:, 0, :] ** 2 + X_iq[:, 1, :] ** 2, axis=1, keepdims=True)
        power = np.maximum(power, 1e-10)
        scale = np.sqrt(power)[:, np.newaxis, :]  # (N, 1, 1)
        return X_iq / scale

    def run(self):
        if not self.file_label_pairs:
            self.finished.emit("")
            return

        # Load all samples
        # Load all samples concurrently
        print(f"Loading {len(self.file_label_pairs)} samples...")
        start_load = time.time()
        
        X_list = [None] * len(self.file_label_pairs)
        y_list = [None] * len(self.file_label_pairs)
        
        def load_one(idx, path, label):
            arr = self._load_array(path)
            if arr is not None:
                arr = np.asarray(arr)
                # Multi-channel arrays (num_rx_ant, N) are kept as-is
                if arr.ndim == 2 and arr.shape[0] <= 64:
                    return idx, arr, label
                return idx, arr.ravel(), label
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_one, i, p, l) for i, (p, l) in enumerate(self.file_label_pairs)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    i, x, y_val = res
                    X_list[i] = x
                    y_list[i] = y_val
        
        # Filter Nones — keep X and y paired so indices stay aligned
        paired = [(x, y) for x, y in zip(X_list, y_list) if x is not None]
        X_list = [p[0] for p in paired]
        y_list = [p[1] for p in paired]
        
        print(f"Loaded {len(X_list)} samples in {time.time() - start_load:.2f}s")

        if not X_list:
            self.finished.emit("")
            return

        # Detect multi-channel arrays (2D with small first dim)
        is_multi_channel = any(
            isinstance(a, np.ndarray) and a.ndim == 2 for a in X_list
        )

        # Use fixed target length to match notebook preprocessing (2048)
        # Raw waveform files can be very long (e.g. 98304); passing them
        # untruncated makes convolutions ~24x slower than intended.
        target_len = self.TARGET_LENGTH
        if target_len <= 0:
            if is_multi_channel:
                target_len = max([a.shape[-1] for a in X_list])
            else:
                target_len = max([a.size for a in X_list])
        if target_len == 0:
            self.finished.emit("")
            return

        if is_multi_channel:
            # Multi-channel: stack as (batch, num_channels, target_len)
            num_ch = X_list[0].shape[0]
            X = np.zeros((len(X_list), num_ch, target_len), dtype=np.float32)
            for i, a in enumerate(X_list):
                if a.ndim == 1:
                    # Mixed: treat 1D as single-channel, replicate
                    L = min(len(a), target_len)
                    for c in range(num_ch):
                        X[i, c, :L] = a[:L].astype(np.float32)
                else:
                    L = min(a.shape[-1], target_len)
                    X[i, :, :L] = a[:, :L].astype(np.float32)
            input_channels = num_ch
            print(f"Multi-channel mode: {num_ch} channels, {target_len} samples")
        else:
            # Pad/truncate to target length
            X = np.zeros((len(X_list), target_len), dtype=np.float32)
            for i, a in enumerate(X_list):
                L = min(len(a), target_len)
                X[i, :L] = a[:L].astype(np.float32)

        y = np.asarray(y_list, dtype=np.int64)

        # Shape data based on model type (only for non-multi-channel)
        if is_multi_channel:
            pass  # already shaped as (batch, num_ch, target_len)
        elif self._is_iq_model:
            X = self._prepare_iq_data(X)
            X = self._normalize_iq(X)
            input_channels = 2
        else:
            X = self._prepare_1ch_data(X)
            input_channels = 1

        # Shuffle
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Split train/val
        split = int(len(X) * (1.0 - self.val_split))
        if split < 1:
            split = max(1, len(X) - 1)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Convert to tensors (keep on CPU initially for pinned memory transfer)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)

        # Create DataLoaders with pin_memory if using GPU
        use_pin = (self.device.type == 'cuda')
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  pin_memory=use_pin)

        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                pin_memory=use_pin)

        # Build model
        num_classes = len(self.labels)
        signal_len = X.shape[2]  # length after channel split
        model = get_model(self.model_name, num_classes=num_classes, input_size=signal_len,
                          in_channels=input_channels, **self.model_hparams)
        model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Scheduler & Scaler
        warmup_epochs = min(3, max(1, int(self.epochs * 0.15)))
        use_amp = (self.device.type == 'cuda')
        scaler = GradScaler('cuda', enabled=use_amp)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.epochs - warmup_epochs),
            eta_min=self.lr * 0.01
        )

        # Training loop
        try:
            for epoch in range(self.epochs):
                if self._stop:
                    print("Training stopped by user.")
                    break

                # Warmup logic
                if epoch < warmup_epochs:
                    curr_lr = self.lr * (epoch + 1) / warmup_epochs
                    for pg in optimizer.param_groups:
                        pg['lr'] = curr_lr
                else:
                    # Scheduler step is usually done at end of epoch, but notebook did it at start
                    # We'll follow notebook logic or standard PyTorch pattern (step at end usually)
                    # Notebook logic: else: scheduler.step() inside loop
                    scheduler.step()

                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for X_batch, y_batch in train_loader:
                    # Non-blocking transfer
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    
                    # Mixed Precision Forward
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    # Mixed Precision Backward
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item() * X_batch.size(0)
                    _, pred = outputs.max(1)
                    train_correct += pred.eq(y_batch).sum().item()
                    train_total += y_batch.size(0)

                train_loss /= max(train_total, 1)
                train_acc = train_correct / max(train_total, 1)

                # Validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        y_batch = y_batch.to(self.device, non_blocking=True)
                        
                        with autocast('cuda', enabled=use_amp):
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            
                        val_loss += loss.item() * X_batch.size(0)
                        _, pred = outputs.max(1)
                        val_correct += pred.eq(y_batch).sum().item()
                        val_total += y_batch.size(0)

                val_loss /= max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                # Emit progress
                self.progress.emit(epoch + 1, self.epochs, train_loss, val_loss, train_acc, val_acc)

        except Exception as e:
            print(f"Training failed: {e}")

        # Save model + metadata
        out_dir = self.save_dir if self.save_dir else os.path.join(os.getcwd(), 'models')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = int(time.time())
        save_path = os.path.join(out_dir, f"{self.model_name}_{timestamp}.pth")
        meta_path = os.path.join(out_dir, f"{self.model_name}_{timestamp}.json")
        try:
            torch.save(model.state_dict(), save_path)
            # Save companion metadata
            metadata = {
                "model_name": self.model_name,
                "class_labels": self.labels,
                "num_classes": num_classes,
                "input_channels": input_channels,
                "signal_length": signal_len,
                "timestamp": timestamp,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save model: {e}")
            save_path = ""

        self.finished.emit(save_path)