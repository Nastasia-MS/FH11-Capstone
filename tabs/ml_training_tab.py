from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QGridLayout, QFrame, QProgressBar)
from PySide6.QtCore import Qt

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
        
        # Model Architecture
        arch_label = QLabel("Model Architecture")
        layout.addWidget(arch_label)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "CNN (Convolutional Neural Network)",
            "RNN (Recurrent Neural Network)",
            "LSTM (Long Short-Term Memory)",
            "Transformer"
        ])
        layout.addWidget(self.arch_combo)
        
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
        layout.addWidget(self.train_btn)
        
        # Status label
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setProperty("class", "stat-label")
        layout.addWidget(self.status_label)
        
        return panel
    
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
        
        self.status_label.setText("Training...")
        self.train_btn.setEnabled(False)
        
        # Placeholder: would connect to actual training logic
        print("Training started...")
        
        # Example: simulate some progress
        # In real implementation, connect to actual training signals/callbacks
        self.progress_bar.setValue(0)
        self.progress_value.setText("Epoch 0/10")
    
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