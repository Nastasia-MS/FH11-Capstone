from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QGridLayout, QFrame, QProgressBar, QFileDialog, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt

import os
import numpy as np

from widgets.training_chart import TrainingChartWidget

#TODO: Add import data input to ML

class MLTrainingTab(QWidget):
    """ML Training configuration and visualization tab"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
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
        """Create the training configuration panel"""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        #layout.setSpacing(20)
        
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
        # Provide a few TF-ready model options with default params
        self.model_combo.addItems(["SimpleCNN", "TinyConv", "MLP"])
        layout.addWidget(self.model_combo)

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
        self.batch_spin.setValue(32)
        epoch_layout.addWidget(self.batch_spin)

        layout.addLayout(epoch_layout)

        # store selected datasets: label -> list[filepaths]
        self.datasets = {}

        # trainer reference
        self._trainer = None

        
        # Learning Rate
        lr_header = QHBoxLayout()
        lr_label = QLabel("Learning Rate")
        self.lr_value = QLabel("0.0010")
        self.lr_value.setProperty("class", "stat-value")
        lr_header.addWidget(lr_label)
        #lr_header.addStretch()
        lr_header.addWidget(self.lr_value)
        layout.addLayout(lr_header)
        
        # Training Parameters
        #layout.addSpacing(10)
        params_layout = QGridLayout()
        #params_layout.setSpacing(12)
        
        self.params = {
            "Training Samples": "8,000",
            "Validation Samples": "2,000",
            "Batch Size": "32",
            "Epochs": "10"
        }
        
        for i, (label, value) in enumerate(self.params.items()):
            label_widget = QLabel(label)
            label_widget.setProperty("class", "stat-label")
            value_widget = QLabel(value)
            value_widget.setProperty("class", "stat-value")
            value_widget.setAlignment(Qt.AlignRight)
            params_layout.addWidget(label_widget, i, 0)
            params_layout.addWidget(value_widget, i, 1)
        
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
        
        return panel

    def add_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", os.path.expanduser("~"))
        if not folder:
            return
        label = os.path.basename(folder.rstrip(os.sep)) or folder
        # ensure unique label
        orig_label = label
        i = 1
        while label in self.datasets:
            label = f"{orig_label}_{i}"
            i += 1

        files = self._gather_dataset_files(folder)
        if not files:
            self.status_label.setText("No supported files in selected folder")
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
        self.train_btn.setEnabled(len(valid_classes) >= 2)

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
            self.loss_chart = TrainingChartWidget("Loss", "Loss")
            layout.addWidget(self.loss_chart)
            
            # Legend
            legend = self.create_legend(
                [("→ Training Loss", "#3b82f6"), ("→ Validation Loss", "#fb923c")]
            )
        else:  # accuracy
            self.acc_chart = TrainingChartWidget("Accuracy", "Accuracy (%)")
            layout.addWidget(self.acc_chart)
            
            # Legend
            legend = self.create_legend(
                [("→ Training Accuracy", "#10b981"), ("→ Validation Accuracy", "#a855f7")]
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

        # start TrainerThread
        model_name = self.model_combo.currentText()
        epochs = int(self.epochs_spin.value())
        batch_size = int(self.batch_spin.value())

        self._trainer = TrainerThread(file_label_pairs, labels, model_name=model_name, epochs=epochs, batch_size=batch_size)
        self._trainer.progress.connect(self.update_training_progress)
        self._trainer.finished.connect(self.on_training_finished)

        self.status_label.setText("Training...")
        self.train_btn.setEnabled(False)
        self._trainer.start()
        print(f"Trainer started: model={model_name}, epochs={epochs}, batch_size={batch_size}")
    
    def update_training_progress(self, epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
        """Update the training progress and charts
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_loss: Training loss value
            val_loss: Validation loss value
            train_acc: Training accuracy value
            val_acc: Validation accuracy value
        """
        self.progress_bar.setValue(epoch)
        self.progress_value.setText(f"Epoch {epoch}/{total_epochs}")
        
        # Update loss chart
        self.loss_chart.add_data_point(train_loss, val_loss)
        
        # Update accuracy chart
        self.acc_chart.add_data_point(train_acc, val_acc)
        
        if epoch >= total_epochs:
            self.status_label.setText("Training Complete")
            self.train_btn.setEnabled(True)
            self.train_btn.setText("▶  Start Training")

    def on_training_finished(self, model_path):
        if model_path:
            self.status_label.setText(f"Training finished — saved: {model_path}")
            print(f"Model saved to: {model_path}")
        else:
            self.status_label.setText("Training finished (no model saved)")
        self.train_btn.setEnabled(True)
        # Properly quit and wait for thread to finish
        if self._trainer:
            self._trainer.quit()
            self._trainer.wait()
            self._trainer = None