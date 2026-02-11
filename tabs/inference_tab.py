from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QGridLayout, QFrame, QFileDialog, QTabWidget)
from PySide6.QtCore import Qt
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import torch


class InferenceResultsTab(QWidget):
    """Model inference and evaluation tab"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.eval_data = None
        self.eval_labels = None
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        title = QLabel("📊 Model Evaluation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Load a trained model and evaluate on test data")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self.load_model)
        control_layout.addWidget(self.load_model_btn)
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setProperty("class", "stat-label")
        control_layout.addWidget(self.model_label)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Data panel
        data_layout = QHBoxLayout()
        
        self.load_data_btn = QPushButton("Load Test Data Folder")
        self.load_data_btn.clicked.connect(self.load_test_data)
        self.load_data_btn.setEnabled(False)
        data_layout.addWidget(self.load_data_btn)
        
        self.data_label = QLabel("No test data loaded")
        self.data_label.setProperty("class", "stat-label")
        data_layout.addWidget(self.data_label)
        
        data_layout.addStretch()
        layout.addLayout(data_layout)
        
        # Evaluation tabs
        self.eval_tabs = QTabWidget()
        
        # Confusion Matrix tab
        cm_widget = self.create_confusion_matrix_widget()
        self.eval_tabs.addTab(cm_widget, "Confusion Matrix")
        
        # Classification Report tab
        report_widget = self.create_report_widget()
        self.eval_tabs.addTab(report_widget, "Classification Report")
        
        # ROC Curve tab
        roc_widget = self.create_roc_widget()
        self.eval_tabs.addTab(roc_widget, "ROC Curve")
        
        self.eval_tabs.setEnabled(False)
        layout.addWidget(self.eval_tabs)
    
    def create_confusion_matrix_widget(self):
        """Create confusion matrix visualization widget"""
        widget = QFrame()
        widget.setObjectName("card")
        layout = QVBoxLayout(widget)
        
        self.cm_figure = Figure(figsize=(6, 5), dpi=100)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        layout.addWidget(self.cm_canvas)
        
        eval_btn_layout = QHBoxLayout()
        self.eval_cm_btn = QPushButton("Evaluate Confusion Matrix")
        self.eval_cm_btn.clicked.connect(self.evaluate_confusion_matrix)
        self.eval_cm_btn.setEnabled(False)
        eval_btn_layout.addWidget(self.eval_cm_btn)
        eval_btn_layout.addStretch()
        layout.addLayout(eval_btn_layout)
        
        return widget
    
    def create_report_widget(self):
        """Create classification report widget"""
        widget = QFrame()
        widget.setObjectName("card")
        layout = QVBoxLayout(widget)
        
        self.report_label = QLabel("Run evaluation to see classification report")
        self.report_label.setProperty("class", "stat-label")
        self.report_label.setWordWrap(True)
        layout.addWidget(self.report_label)
        
        eval_btn_layout = QHBoxLayout()
        self.eval_report_btn = QPushButton("Generate Report")
        self.eval_report_btn.clicked.connect(self.evaluate_report)
        self.eval_report_btn.setEnabled(False)
        eval_btn_layout.addWidget(self.eval_report_btn)
        eval_btn_layout.addStretch()
        layout.addLayout(eval_btn_layout)
        
        return widget
    
    def create_roc_widget(self):
        """Create ROC curve visualization widget"""
        widget = QFrame()
        widget.setObjectName("card")
        layout = QVBoxLayout(widget)
        
        self.roc_figure = Figure(figsize=(6, 5), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        layout.addWidget(self.roc_canvas)
        
        eval_btn_layout = QHBoxLayout()
        self.eval_roc_btn = QPushButton("Evaluate ROC Curve")
        self.eval_roc_btn.clicked.connect(self.evaluate_roc)
        self.eval_roc_btn.setEnabled(False)
        eval_btn_layout.addWidget(self.eval_roc_btn)
        eval_btn_layout.addStretch()
        layout.addLayout(eval_btn_layout)
        
        return widget
    
    def load_model(self):
        """Load a trained PyTorch model"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", os.path.expanduser("~"),
            "PyTorch Models (*.pth);;All Files (*)"
        )
        if not filepath:
            return
        
        try:
            from backend.torch_models import get_model
            # Try to load state dict
            state_dict = torch.load(filepath, map_location='cpu')
            # We don't know the num_classes yet, so use a default and update if needed
            self.model = get_model('SimpleCNN', num_classes=2)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_path = filepath
            self.model_label.setText(f"Loaded: {os.path.basename(filepath)}")
            self.load_data_btn.setEnabled(True)
        except Exception as e:
            self.model_label.setText(f"Failed to load: {e}")
            print(f"Error loading model: {e}")
    
    def load_test_data(self):
        """Load test data from a folder (same structure as training)"""
        folder = QFileDialog.getExistingDirectory(self, "Select Test Data Folder")
        if not folder:
            return
        
        try:
            # Gather files
            exts = ('.npy', '.npz', '.csv')
            files = []
            for entry in os.listdir(folder):
                path = os.path.join(folder, entry)
                if os.path.isfile(path) and entry.lower().endswith(exts):
                    files.append(path)
            
            if not files:
                self.data_label.setText("No supported files found")
                return
            
            # Load all samples
            X_list, y_list = [], []
            for i, path in enumerate(files):
                arr = self._load_array(path)
                if arr is None:
                    continue
                arr = np.asarray(arr).ravel()
                X_list.append(arr)
                y_list.append(i % 2)  # Simple binary classification fallback
            
            if not X_list:
                self.data_label.setText("Failed to load any files")
                return
            
            max_len = max([a.size for a in X_list])
            X = np.zeros((len(X_list), max_len), dtype=np.float32)
            for i, a in enumerate(X_list):
                L = min(len(a), max_len)
                X[i, :L] = a[:L]
            
            X = X[..., np.newaxis]
            X = np.transpose(X, (0, 2, 1))
            
            self.eval_data = torch.from_numpy(X)
            self.eval_labels = np.asarray(y_list, dtype=np.int64)
            
            self.data_label.setText(f"Loaded {len(files)} samples")
            self.eval_cm_btn.setEnabled(True)
            self.eval_report_btn.setEnabled(True)
            self.eval_roc_btn.setEnabled(True)
            self.eval_tabs.setEnabled(True)
        except Exception as e:
            self.data_label.setText(f"Error loading data: {e}")
            print(f"Error: {e}")
    
    def _load_array(self, path):
        """Load array from file"""
        try:
            if path.lower().endswith('.npy') or path.lower().endswith('.npz'):
                arr = np.load(path, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    keys = list(arr.keys())
                    arr = arr[keys[0]] if keys else None
                return np.asarray(arr, dtype=np.float32)
            if path.lower().endswith('.csv'):
                return np.loadtxt(path, delimiter=',').astype(np.float32)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
        return None
    
    def evaluate_confusion_matrix(self):
        """Compute and display confusion matrix"""
        if self.model is None or self.eval_data is None:
            return
        
        try:
            with torch.no_grad():
                outputs = self.model(self.eval_data)
                _, predictions = outputs.max(1)
            
            y_pred = predictions.cpu().numpy()
            cm = confusion_matrix(self.eval_labels, y_pred)
            
            self.cm_figure.clear()
            ax = self.cm_figure.add_subplot(111)
            im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
            
            self.cm_figure.colorbar(im, ax=ax)
            self.cm_canvas.draw()
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
    
    def evaluate_report(self):
        """Compute and display classification report"""
        if self.model is None or self.eval_data is None:
            return
        
        try:
            with torch.no_grad():
                outputs = self.model(self.eval_data)
                _, predictions = outputs.max(1)
            
            y_pred = predictions.cpu().numpy()
            report = classification_report(self.eval_labels, y_pred, zero_division=0)
            accuracy = (y_pred == self.eval_labels).mean()
            
            text = f"Accuracy: {accuracy:.4f}\n\n{report}"
            self.report_label.setText(text)
        except Exception as e:
            self.report_label.setText(f"Error: {e}")
            print(f"Error: {e}")
    
    def evaluate_roc(self):
        """Compute and display ROC curve (binary classification only)"""
        if self.model is None or self.eval_data is None:
            return
        
        try:
            with torch.no_grad():
                outputs = self.model(self.eval_data)
                # Get probabilities for class 1
                if outputs.shape[1] == 2:
                    probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    probs = outputs[:, 0].cpu().numpy()
            
            fpr, tpr, _ = roc_curve(self.eval_labels, probs)
            roc_auc = auc(fpr, tpr)
            
            self.roc_figure.clear()
            ax = self.roc_figure.add_subplot(111)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            
            self.roc_canvas.draw()
        except Exception as e:
            print(f"Error computing ROC curve: {e}")