from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QGridLayout, QFrame, QProgressBar, QFileDialog,
                               QListWidget, QListWidgetItem, QGroupBox, QDoubleSpinBox, QScrollArea)
from PySide6.QtCore import Qt, Signal, QSettings

import os
import numpy as np
import torch

from widgets.training_chart import TrainingChartWidget


class MLTrainingTab(QWidget):
    """ML Training configuration and visualization tab"""
    # Emitted when a model finishes training: (model_path, class_labels_list)
    trained_model_ready = Signal(str, list)

    def __init__(self, dataset_manager=None):
        super().__init__()
        self.dataset_manager = dataset_manager
        
        self.setup_ui()
        if self.dataset_manager is not None:
            self.load_registry_btn.setEnabled(True)
        
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - Training Configuration
        left_panel = self.create_configuration_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel - Charts
        right_panel = self.create_charts_panel()
        layout.addWidget(right_panel, 2)
    
    def create_configuration_panel(self):
        """Create the training configuration panel (scrollable)"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title = QLabel("⚙️ Training Configuration")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Configure ML model parameters")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        # Dataset selection (support multiple folders / classes)
        data_btn_layout = QHBoxLayout()
        self.add_data_btn = QPushButton("Add Data Folder")
        self.add_data_btn.clicked.connect(self.add_data_folder)
        data_btn_layout.addWidget(self.add_data_btn)

        self.load_registry_btn = QPushButton("📂 Load from Datasets")
        self.load_registry_btn.setToolTip("Group datasets by modulation type and load as training classes")
        self.load_registry_btn.clicked.connect(self.load_from_registry)
        self.load_registry_btn.setEnabled(False)
        data_btn_layout.addWidget(self.load_registry_btn)

        self.quick_load_btn = QPushButton("📦 Quick Load Dataset")
        self.quick_load_btn.setToolTip("Auto-load class folders from waveform_data/ directory")
        self.quick_load_btn.clicked.connect(self.quick_load_dataset)
        data_btn_layout.addWidget(self.quick_load_btn)

        self.remove_data_btn = QPushButton("Remove Selected")
        self.remove_data_btn.clicked.connect(self.remove_selected_dataset)
        self.remove_data_btn.setEnabled(False)
        data_btn_layout.addWidget(self.remove_data_btn)

        self.clear_data_btn = QPushButton("Clear All")
        self.clear_data_btn.clicked.connect(self.clear_datasets)
        self.clear_data_btn.setEnabled(False)
        data_btn_layout.addWidget(self.clear_data_btn)

        layout.addLayout(data_btn_layout)

        # list of selected dataset folders (each folder => a class)
        self.dataset_list = QListWidget()
        self.dataset_list.itemSelectionChanged.connect(self.on_dataset_selection_changed)
        layout.addWidget(self.dataset_list)
        
        # Model Architecture
        arch_label = QLabel("Model Architecture")
        layout.addWidget(arch_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SimpleCNN", "TinyConv", "MLP", "ResNet1DOptimized"])
        layout.addWidget(self.model_combo)

        # Model save path (optional override)
        from PySide6.QtWidgets import QLineEdit
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save model to:"))
        self.model_save_path_edit = QLineEdit()
        self.model_save_path_edit.setPlaceholderText("Default: models/ folder")
        self.model_save_path_edit.setReadOnly(True)
        save_row.addWidget(self.model_save_path_edit, 1)
        browse_save_btn = QPushButton("Browse…")
        browse_save_btn.clicked.connect(self._browse_save_path)
        save_row.addWidget(browse_save_btn)
        layout.addLayout(save_row)

        # Epochs and batch size controls
        epoch_layout = QHBoxLayout()
        epoch_label = QLabel("Epochs")
        epoch_layout.addWidget(epoch_label)
        from PySide6.QtWidgets import QSpinBox
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(10)
        epoch_layout.addWidget(self.epochs_spin)

        batch_label = QLabel("Batch Size")
        epoch_layout.addWidget(batch_label)
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(1024)
        self.batch_spin.setValue(64)
        epoch_layout.addWidget(self.batch_spin)

        layout.addLayout(epoch_layout)

        # store selected datasets: label -> list[filepaths]
        self.datasets = {}

        # trainer reference
        self._trainer = None

        # --- Training Hyperparameters (collapsible) ---
        self.training_hparams_group = QGroupBox("Training Hyperparameters")
        self.training_hparams_group.setCheckable(True)
        self.training_hparams_group.setChecked(False)
        th_layout = QGridLayout(self.training_hparams_group)
        th_layout.setSpacing(6)

        # Learning Rate
        th_layout.addWidget(QLabel("Learning Rate"), 0, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        th_layout.addWidget(self.lr_spin, 0, 1)

        # Weight Decay
        th_layout.addWidget(QLabel("Weight Decay"), 1, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(1e-4)
        th_layout.addWidget(self.weight_decay_spin, 1, 1)

        # Label Smoothing
        th_layout.addWidget(QLabel("Label Smoothing"), 2, 0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 1.0)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setSingleStep(0.05)
        self.label_smoothing_spin.setValue(0.1)
        th_layout.addWidget(self.label_smoothing_spin, 2, 1)

        # Validation Split
        th_layout.addWidget(QLabel("Validation Split"), 3, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        th_layout.addWidget(self.val_split_spin, 3, 1)

        # Gradient Clip
        th_layout.addWidget(QLabel("Gradient Clip"), 4, 0)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 100.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSingleStep(0.1)
        self.grad_clip_spin.setValue(1.0)
        th_layout.addWidget(self.grad_clip_spin, 4, 1)

        # Hide contents initially (collapsed)
        for i in range(th_layout.count()):
            w = th_layout.itemAt(i).widget()
            if w:
                w.setVisible(False)
        self.training_hparams_group.toggled.connect(
            lambda checked: self._toggle_group(self.training_hparams_group, checked))
        layout.addWidget(self.training_hparams_group)

        # --- Per-Model Hyperparameters (collapsible) ---
        self.model_hparams_group = QGroupBox("Model Hyperparameters")
        self.model_hparams_group.setCheckable(True)
        self.model_hparams_group.setChecked(False)
        self._mh_layout = QGridLayout(self.model_hparams_group)
        self._mh_layout.setSpacing(6)

        # ResNet1DOptimized params
        self._mh_base_filters_label = QLabel("Base Filters")
        self._mh_layout.addWidget(self._mh_base_filters_label, 0, 0)
        self.resnet_base_filters_spin = QDoubleSpinBox()
        self.resnet_base_filters_spin.setRange(8, 512)
        self.resnet_base_filters_spin.setDecimals(0)
        self.resnet_base_filters_spin.setSingleStep(8)
        self.resnet_base_filters_spin.setValue(64)
        self._mh_layout.addWidget(self.resnet_base_filters_spin, 0, 1)

        self._mh_dropout_label = QLabel("Dropout")
        self._mh_layout.addWidget(self._mh_dropout_label, 1, 0)
        self.resnet_dropout_spin = QDoubleSpinBox()
        self.resnet_dropout_spin.setRange(0.0, 0.9)
        self.resnet_dropout_spin.setDecimals(2)
        self.resnet_dropout_spin.setSingleStep(0.05)
        self.resnet_dropout_spin.setValue(0.2)
        self._mh_layout.addWidget(self.resnet_dropout_spin, 1, 1)

        self._mh_no_params_label = QLabel("No editable hyperparameters for this model.")
        self._mh_layout.addWidget(self._mh_no_params_label, 0, 0, 1, 2)

        # Hide all initially
        for i in range(self._mh_layout.count()):
            w = self._mh_layout.itemAt(i).widget()
            if w:
                w.setVisible(False)
        self.model_hparams_group.toggled.connect(
            lambda checked: self._on_model_hparams_toggled(checked))
        self.model_combo.currentTextChanged.connect(self._update_model_hparams_visibility)
        layout.addWidget(self.model_hparams_group)

        # Training Parameters
        params_layout = QGridLayout()
        
        # We define these as class attributes so we can access them in update_training_progress
        self.val_labels = {
            "Training Samples": QLabel("0"),
            "Validation Samples": QLabel("0"),
            "Batch Size": QLabel(str(self.batch_spin.value())),
            "Epochs": QLabel(str(self.epochs_spin.value()))
        }
        
        for i, (label, widget) in enumerate(self.val_labels.items()):
            label_widget = QLabel(label)
            label_widget.setProperty("class", "stat-label")
            
            widget.setProperty("class", "stat-value")
            widget.setAlignment(Qt.AlignRight)
            
            params_layout.addWidget(label_widget, i, 0)
            params_layout.addWidget(widget, i, 1)
            
        # Connect changes in input to the display labels immediately
        self.batch_spin.valueChanged.connect(lambda v: self.val_labels["Batch Size"].setText(str(v)))
        self.epochs_spin.valueChanged.connect(lambda v: self.val_labels["Epochs"].setText(str(v)))

        layout.addLayout(params_layout)
        
        # Progress section
        #layout.addSpacing(10)
        progress_header = QHBoxLayout()
        progress_label = QLabel("Training Progress")
        progress_label.setProperty("class", "stat-label")
        self.progress_value = QLabel("Epoch 0/10")
        self.progress_value.setProperty("class", "stat-value")
        progress_header.addWidget(progress_label)
        #progress_header.addStretch()
        progress_header.addWidget(self.progress_value)
        layout.addLayout(progress_header)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(10)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        #layout.addStretch()
        
        # Start Training button
        self.train_btn = QPushButton("▶  Start Training")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.clicked.connect(self.start_training)
        # disable until dataset selected
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)
        
        # Status label
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setProperty("class", "stat-label")
        layout.addWidget(self.status_label)
        
        # Wrap in scroll area so the panel doesn't compress
        scroll = QScrollArea()
        scroll.setObjectName("card")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    def _toggle_group(self, group, checked):
        """Show/hide all child widgets inside a QGroupBox."""
        group_layout = group.layout()
        for i in range(group_layout.count()):
            w = group_layout.itemAt(i).widget()
            if w:
                w.setVisible(checked)
        # For the model hparams group, also apply model-specific visibility
        if group is self.model_hparams_group and checked:
            self._update_model_hparams_visibility(self.model_combo.currentText())

    def _on_model_hparams_toggled(self, checked):
        """Handle model hyperparameters group toggle."""
        self._toggle_group(self.model_hparams_group, checked)

    def _update_model_hparams_visibility(self, model_name):
        """Show/hide model-specific hyperparameter widgets based on selected model."""
        if not self.model_hparams_group.isChecked():
            return
        is_resnet = model_name == 'ResNet1DOptimized'
        self._mh_base_filters_label.setVisible(is_resnet)
        self.resnet_base_filters_spin.setVisible(is_resnet)
        self._mh_dropout_label.setVisible(is_resnet)
        self.resnet_dropout_spin.setVisible(is_resnet)
        self._mh_no_params_label.setVisible(not is_resnet)

    def add_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", os.path.expanduser("~"))
        if not folder:
            return

        # Check if folder contains class subfolders (each subfolder = a class)
        subdirs = [d for d in sorted(os.listdir(folder))
                   if os.path.isdir(os.path.join(folder, d))]
        subdir_with_files = []
        for d in subdirs:
            files = self._gather_dataset_files(os.path.join(folder, d))
            if files:
                subdir_with_files.append((d, files))

        if subdir_with_files:
            # Auto-add each subfolder as a separate class
            for sub_label, files in subdir_with_files:
                label = sub_label
                orig_label = label
                i = 1
                while label in self.datasets:
                    label = f"{orig_label}_{i}"
                    i += 1
                self.datasets[label] = files
                item = QListWidgetItem(f"{label} ({len(files)} files)")
                item.setData(Qt.UserRole, label)
                self.dataset_list.addItem(item)
            self.clear_data_btn.setEnabled(True)
            self.status_label.setText(f"Loaded {len(subdir_with_files)} classes from {os.path.basename(folder)}")
            self._update_train_button_state()
            return

        # Fallback: treat folder itself as a single class
        label = os.path.basename(folder.rstrip(os.sep)) or folder
        orig_label = label
        i = 1
        while label in self.datasets:
            label = f"{orig_label}_{i}"
            i += 1

        files = self._gather_dataset_files(folder)
        if not files:
            self.status_label.setText("No supported files (.npy/.npz/.csv) found")
            return

        self.datasets[label] = files
        item = QListWidgetItem(f"{label} ({len(files)} files)")
        item.setData(Qt.UserRole, label)
        self.dataset_list.addItem(item)
        self.clear_data_btn.setEnabled(True)
        self._update_train_button_state()

    def remove_selected_dataset(self):
        items = self.dataset_list.selectedItems()
        if not items:
            return
        for it in items:
            label = it.data(Qt.UserRole)
            if label in self.datasets:
                del self.datasets[label]
            row = self.dataset_list.row(it)
            self.dataset_list.takeItem(row)
        self.remove_data_btn.setEnabled(False)
        if self.dataset_list.count() == 0:
            self.clear_data_btn.setEnabled(False)
        self._update_train_button_state()

    def clear_datasets(self):
        self.datasets.clear()
        self.dataset_list.clear()
        self.remove_data_btn.setEnabled(False)
        self.clear_data_btn.setEnabled(False)
        self._update_train_button_state()

    def on_dataset_selection_changed(self):
        self.remove_data_btn.setEnabled(len(self.dataset_list.selectedItems()) > 0)

    def _update_train_button_state(self):
        # enable training only when there are at least two classes with files
        valid_classes = [k for k, v in self.datasets.items() if v]
        can_train = len(valid_classes) >= 2
        self.train_btn.setEnabled(can_train)
        if valid_classes and not can_train:
            self.status_label.setText(
                f"Need at least 2 classes to train (currently {len(valid_classes)}). "
                "Add more data folders or use Batch Generate on the Waveform tab."
            )

    def _gather_dataset_files(self, folder):
        """Return list of candidate data files in folder (.npy, .npz, .csv)."""
        exts = ('.npy', '.npz', '.csv')
        files = []
        try:
            for entry in os.listdir(folder):
                path = os.path.join(folder, entry)
                if os.path.isfile(path) and entry.lower().endswith(exts):
                    files.append(path)
        except Exception as e:
            print(f"Error listing dataset folder: {e}")
        return sorted(files)

    def _load_sample_from_file(self, path):
        """Load a sample from file. Supports .npy/.npz/.csv."""
        try:
            if path.lower().endswith('.npy'):
                data = np.load(path, allow_pickle=True)
                return data
            if path.lower().endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                # return first array inside
                if isinstance(data, np.lib.npyio.NpzFile):
                    keys = list(data.keys())
                    return data[keys[0]] if keys else None
            if path.lower().endswith('.csv'):
                data = np.loadtxt(path, delimiter=',')
                return data
        except Exception as e:
            print(f"Failed to load sample {path}: {e}")
        return None
    
    def create_charts_panel(self):
        """Create the charts visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Loss Chart
        loss_card = self.create_chart_card(
            "Training & Validation Loss",
            "Loss convergence over epochs",
            "loss"
        )
        layout.addWidget(loss_card)
        
        # Accuracy Chart
        acc_card = self.create_chart_card(
            "Training & Validation Accuracy",
            "Classification accuracy over epochs",
            "accuracy"
        )
        layout.addWidget(acc_card)
        
        return panel
    
    def create_chart_card(self, title, subtitle, chart_type):
        """Create a chart card with title and legend"""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        title_label = QLabel(title)
        title_label.setProperty("class", "card-title")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setProperty("class", "section-subtitle")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        
        # Chart
        if chart_type == "loss":
            self.loss_chart = TrainingChartWidget("Loss", "Loss", cap_at_one=False)
            layout.addWidget(self.loss_chart)
            
            # Legend
            legend = self.create_legend(
                [("→ Training Loss", "#3b82f6"), ("→ Validation Loss", "#fb923c")]
            )
        else:  # accuracy
            self.acc_chart = TrainingChartWidget("Accuracy", "Accuracy (%)", cap_at_one=True)
            layout.addWidget(self.acc_chart)
            
            # Legend
            legend = self.create_legend(
                [("→ Training Accuracy", "#3b82f6"), ("→ Validation Accuracy", "#fb923c")]
            )
        
        layout.addLayout(legend)
        
        return card
    
    def create_legend(self, items):
        """Create a legend layout for the chart"""
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        
        for i, (text, color) in enumerate(items):
            label = QLabel(text)
            label.setStyleSheet(f"color: {color}; font-size: 12px;")
            legend_layout.addWidget(label)
            if i < len(items) - 1:
                legend_layout.addSpacing(20)
        
        legend_layout.addStretch()
        return legend_layout
    
    def load_from_registry(self):
        """Load datasets from the shared DatasetManager, grouped by modulation as classes."""
        if self.dataset_manager is None:
            return

        entries = self.dataset_manager.scan()
        base_entries = [e for e in entries if not e.get('augmented', False)]

        if not base_entries:
            self.status_label.setText("No datasets in registry — generate some first.")
            return

        from collections import defaultdict
        by_modulation = defaultdict(list)
        for entry in base_entries:
            mod = entry.get('modulation', entry['name'])
            npy = entry.get('_npy_path', '')
            if npy:
                by_modulation[mod].append(npy)

        if len(by_modulation) < 2:
            self.status_label.setText(
                f"Only {len(by_modulation)} modulation class(es) found — need ≥ 2. "
                "Use Batch Generate on the Waveform tab."
            )

        added = 0
        for mod, files in sorted(by_modulation.items()):
            label = mod
            base_label = label
            i = 1
            while label in self.datasets:
                label = f"{base_label}_{i}"
                i += 1
            self.datasets[label] = files
            item = QListWidgetItem(f"{label} ({len(files)} files) [registry]")
            item.setData(Qt.UserRole, label)
            self.dataset_list.addItem(item)
            added += 1

        self.clear_data_btn.setEnabled(True)
        self.status_label.setText(f"Loaded {added} class(es) from registry")
        self._update_train_button_state()

    def _browse_save_path(self):
        """Let the user pick a directory to save the trained model into."""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Save Folder")
        if folder:
            self.model_save_path_edit.setText(folder)

    def quick_load_dataset(self):
        """Auto-load class folders from waveform_data/ directory."""
        # Use script directory rather than CWD to find waveform_data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)  # parent of tabs/
        waveform_dir = os.path.join(base_dir, 'waveform_data')

        # Fallback to CWD if not found relative to script
        if not os.path.isdir(waveform_dir):
            waveform_dir = os.path.join(os.getcwd(), 'waveform_data')

        if not os.path.isdir(waveform_dir):
            self.status_label.setText("No waveform_data/ directory found")
            return

        self.clear_datasets()
        loaded = 0
        for entry in sorted(os.listdir(waveform_dir)):
            class_dir = os.path.join(waveform_dir, entry)
            if not os.path.isdir(class_dir):
                continue
            files = self._gather_dataset_files(class_dir)
            if not files:
                continue
            label = entry
            self.datasets[label] = files
            item = QListWidgetItem(f"{label} ({len(files)} files)")
            item.setData(Qt.UserRole, label)
            self.dataset_list.addItem(item)
            loaded += 1

        if loaded > 0:
            self.clear_data_btn.setEnabled(True)
            self.status_label.setText(f"Loaded {loaded} classes from waveform_data/")
        else:
            self.status_label.setText(
                f"No class subfolders with data found in {waveform_dir}. "
                "Use Batch Generate on the Waveform tab first."
            )
        self._update_train_button_state()

    def start_training(self):
        """Handle training start button click"""
        # Lazy import to avoid breaking app if torch not installed
        try:
            from backend.trainer import TrainerThread
        except ImportError as e:
            self.status_label.setText(f"PyTorch not installed: {e}")
            print(f"ERROR: Could not import TrainerThread: {e}")
            return

        # Require at least two classes (folders) to train a classifier
        if not self.datasets or len(self.datasets) < 2:
            self.status_label.setText("Add at least two data folders (classes) to train")
            return

        # Build file -> label mapping
        labels = list(self.datasets.keys())
        file_label_pairs = []
        for idx, label in enumerate(labels):
            files = self.datasets.get(label, [])
            for f in files:
                file_label_pairs.append((f, idx))

        if not file_label_pairs:
            self.status_label.setText("No supported files found in datasets")
            return
        
        # Read training hyperparameters from UI
        lr = self.lr_spin.value()
        val_split = self.val_split_spin.value()
        weight_decay = self.weight_decay_spin.value()
        label_smoothing = self.label_smoothing_spin.value()
        grad_clip = self.grad_clip_spin.value()

        total_files = len(file_label_pairs)
        train_count = int(total_files * (1.0 - val_split))
        val_count = total_files - train_count

        # Update the UI labels with ACTUAL data numbers
        self.val_labels["Training Samples"].setText(f"{train_count:,}")
        self.val_labels["Validation Samples"].setText(f"{val_count:,}")

        # start TrainerThread
        model_name = self.model_combo.currentText()
        epochs = int(self.epochs_spin.value())
        batch_size = int(self.batch_spin.value())

        # Collect per-model hyperparameters
        model_hparams = {}
        if model_name == 'ResNet1DOptimized':
            model_hparams['base_filters'] = int(self.resnet_base_filters_spin.value())
            model_hparams['dropout'] = self.resnet_dropout_spin.value()

        # Update progress bar max and clear charts
        self.progress_bar.setMaximum(epochs)
        self.progress_bar.setValue(0)
        self.progress_value.setText(f"Epoch 0/{epochs}")
        self.loss_chart.clear_data()
        self.acc_chart.clear_data()

        # Resolve model save directory: UI override → QSettings modelPath → project models/
        from PySide6.QtCore import QSettings
        save_dir = self.model_save_path_edit.text().strip()
        if not save_dir:
            save_dir = QSettings("MyCompany", "MixedSignalGUI").value("modelPath", "")
        if not save_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(os.path.dirname(script_dir), "models")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get current device setting
        settings = QSettings("MyCompany", "MixedSignalGUI")
        compute_mode = settings.value("mode", "CPU")
        device = 'cuda' if (compute_mode == "GPU" and torch.cuda.is_available()) else 'cpu'

        self._trainer = TrainerThread(
            file_label_pairs, labels,
            model_name=model_name, epochs=epochs, batch_size=batch_size,
            lr=lr, val_split=val_split,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            grad_clip=grad_clip, model_hparams=model_hparams,
            save_dir=save_dir,
            device=device,
        )
        self._trainer.progress.connect(self.update_training_progress)
        self._trainer.finished.connect(self.on_training_finished)

        # Store labels for signal emission later
        self._training_labels = labels

        self.status_label.setText("Training...")
        self.train_btn.setEnabled(False)
        self._trainer.start()
        print(f"Trainer started: model={model_name}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
        
    def update_training_progress(self, epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
        """Update the training progress and labels with dynamic values"""
        
        # 1. Update Progress Bar & Text
        self.progress_bar.setValue(epoch)
        self.progress_value.setText(f"Epoch {epoch}/{total_epochs}")
        
        # 2. Update Charts
        self.loss_chart.add_data_point(train_loss, val_loss)
        self.acc_chart.add_data_point(train_acc, val_acc)
        
        # 3. Update the Grid Labels (Dynamic values)
        # We can use the 'Epochs' label to show current/total
        self.val_labels["Epochs"].setText(f"{epoch} / {total_epochs}")
        
        # 4. Update Status Label
        # Multiplying by 100 assuming trainer sends 0.0-1.0 range
        metrics_text = f"Epoch {epoch}: Loss {train_loss:.4f} | Acc {train_acc*100:.1f}%"
        self.status_label.setText(metrics_text)

        # 5. Handle Completion
        if epoch >= total_epochs:
            self.status_label.setText(f"✅ Complete | Final Acc: {val_acc*100:.1f}%")
            self.train_btn.setEnabled(True)
            self.train_btn.setText("▶  Start Training")
    


    def on_training_finished(self, model_path):
        if model_path:
            self.status_label.setText(f"Training finished — saved: {model_path}")
            print(f"Model saved to: {model_path}")
            # Notify inference tab
            labels = getattr(self, '_training_labels', [])
            self.trained_model_ready.emit(model_path, labels)
        else:
            self.status_label.setText("Training finished (no model saved)")
        self.train_btn.setEnabled(True)
        # Properly quit and wait for thread to finish
        if self._trainer:
            self._trainer.quit()
            self._trainer.wait()
            self._trainer = None