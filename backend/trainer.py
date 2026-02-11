from PySide6.QtCore import QThread, Signal
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .torch_models import get_model


class TrainerThread(QThread):
    # Signals: epoch, total_epochs, train_loss, val_loss, train_acc, val_acc
    progress = Signal(int, int, float, float, float, float)
    finished = Signal(str)

    def __init__(self, file_label_pairs, labels, model_name='SimpleCNN', epochs=10, batch_size=32, lr=0.001, val_split=0.2):
        super().__init__()
        self.file_label_pairs = list(file_label_pairs)
        self.labels = list(labels)
        self.model_name = model_name
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_split = float(val_split)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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

    def run(self):
        if not self.file_label_pairs:
            self.finished.emit("")
            return

        # Load all samples
        X_list = []
        y_list = []
        for path, label in self.file_label_pairs:
            arr = self._load_array(path)
            if arr is None:
                continue
            arr = np.asarray(arr).ravel()
            X_list.append(arr)
            y_list.append(label)

        if not X_list:
            self.finished.emit("")
            return

        max_len = max([a.size for a in X_list])
        if max_len == 0:
            self.finished.emit("")
            return

        # Pad/truncate to same length
        X = np.zeros((len(X_list), max_len), dtype=np.float32)
        for i, a in enumerate(X_list):
            L = min(len(a), max_len)
            X[i, :L] = a[:L]

        # Add channel dimension
        X = X[..., np.newaxis]
        y = np.asarray(y_list, dtype=np.int64)

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

        # Convert to tensors
        X_train = torch.from_numpy(X_train).to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        X_val = torch.from_numpy(X_val).to(self.device)
        y_val = torch.from_numpy(y_val).to(self.device)

        # Transpose to (batch, channel, length) for Conv1d
        X_train = X_train.permute(0, 2, 1)
        X_val = X_val.permute(0, 2, 1)

        # Create DataLoaders
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        num_classes = len(self.labels)
        model = get_model(self.model_name, num_classes=num_classes, input_size=max_len)
        model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        try:
            for epoch in range(self.epochs):
                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                    _, pred = outputs.max(1)
                    train_correct += pred.eq(y_batch).sum().item()
                    train_total += y_batch.size(0)

                train_loss /= train_total
                train_acc = train_correct / train_total

                # Validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
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

        # Save model
        out_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{self.model_name}_{int(time.time())}.pth")
        try:
            torch.save(model.state_dict(), save_path)
        except Exception as e:
            print(f"Failed to save model: {e}")
            save_path = ""

        self.finished.emit(save_path)
