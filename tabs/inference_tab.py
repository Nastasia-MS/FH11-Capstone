from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QGridLayout, QFrame, QFileDialog, QTabWidget)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QApplication
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import torch


def _mpl_theme():
    """Get current theme colors from Qt palette."""
    p = QApplication.palette()
    return (p.window().color().name(),
            p.windowText().color().name(),
            p.base().color().name())


def _apply_mpl_theme(fig, *axes):
    """Apply theme colors to matplotlib figure and axes."""
    bg, fg, axes_bg = _mpl_theme()
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(axes_bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(axes_bg)
            for text in leg.get_texts():
                text.set_color(fg)


class InferenceResultsTab(QWidget):
    """Model inference and evaluation tab"""
    
    def __init__(self, dataset_manager=None):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.model = None
        self.model_path = None
        self.class_labels = []   # set from trained_model_ready signal or manual load
        self.eval_data = None
        self.eval_labels = None
        
        self._cm_data = None
        self._roc_data = None
        
        self.setup_ui()
    
    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            bg, _, _ = _mpl_theme()
            self.cm_figure.patch.set_facecolor(bg)
            self.roc_figure.patch.set_facecolor(bg)
            for ax in self.cm_figure.get_axes():
                _apply_mpl_theme(self.cm_figure, ax)
            for ax in self.roc_figure.get_axes():
                _apply_mpl_theme(self.roc_figure, ax)
            self.cm_canvas.draw_idle()
            self.roc_canvas.draw_idle()
            if self._cm_data:
                self._plot_confusion_matrix(*self._cm_data)
            if self._roc_data:
                self._plot_roc_curve(*self._roc_data)
        super().changeEvent(event)
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        title = QLabel("Model Evaluation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Load a trained model and evaluate on test data")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        # ── Model loading card ────────────────────────────────────────────
        model_card = QFrame()
        model_card.setObjectName("card")
        mc_layout = QVBoxLayout(model_card)
        mc_layout.setContentsMargins(16, 12, 16, 12)
        mc_layout.setSpacing(6)

        model_row = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self.load_model)
        model_row.addWidget(self.load_model_btn)
        self.model_label = QLabel("No model loaded")
        self.model_label.setProperty("class", "stat-label")
        model_row.addWidget(self.model_label)
        model_row.addStretch()
        mc_layout.addLayout(model_row)

        labels_row = QHBoxLayout()
        labels_row.addWidget(QLabel("Classes:"))
        self.class_labels_label = QLabel("—")
        self.class_labels_label.setProperty("class", "stat-label")
        labels_row.addWidget(self.class_labels_label)
        labels_row.addStretch()
        mc_layout.addLayout(labels_row)
        layout.addWidget(model_card)

        # ── Test data loading card ────────────────────────────────────────
        data_card = QFrame()
        data_card.setObjectName("card")
        dc_layout = QVBoxLayout(data_card)
        dc_layout.setContentsMargins(16, 12, 16, 12)
        dc_layout.setSpacing(6)

        data_btn_row = QHBoxLayout()
        self.load_data_btn = QPushButton("Load Test Data Folder")
        self.load_data_btn.clicked.connect(self.load_test_data)
        self.load_data_btn.setEnabled(False)
        data_btn_row.addWidget(self.load_data_btn)

        self.load_registry_btn = QPushButton("📂 Load from Datasets Registry")
        self.load_registry_btn.clicked.connect(self.load_from_registry)
        self.load_registry_btn.setEnabled(False)
        data_btn_row.addWidget(self.load_registry_btn)
        data_btn_row.addStretch()
        dc_layout.addLayout(data_btn_row)

        self.data_label = QLabel("No test data loaded")
        self.data_label.setProperty("class", "stat-label")
        dc_layout.addWidget(self.data_label)
        layout.addWidget(data_card)
        
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
        
        bg, _, _ = _mpl_theme()
        self.cm_figure = Figure(figsize=(6, 5), dpi=100, facecolor=bg)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        self.cm_canvas.setStyleSheet("background: transparent;")
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
        
        bg, _, _ = _mpl_theme()
        self.roc_figure = Figure(figsize=(6, 5), dpi=100, facecolor=bg)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        self.roc_canvas.setStyleSheet("background: transparent;")
        layout.addWidget(self.roc_canvas)
        
        eval_btn_layout = QHBoxLayout()
        self.eval_roc_btn = QPushButton("Evaluate ROC Curve")
        self.eval_roc_btn.clicked.connect(self.evaluate_roc)
        self.eval_roc_btn.setEnabled(False)
        eval_btn_layout.addWidget(self.eval_roc_btn)
        eval_btn_layout.addStretch()
        layout.addLayout(eval_btn_layout)
        
        return widget
    
    def on_trained_model_ready(self, model_path: str, labels: list):
        """Slot connected to MLTrainingTab.trained_model_ready.
        Auto-loads the freshly trained model and stores class labels."""
        self.class_labels = labels
        self.class_labels_label.setText(", ".join(labels) if labels else "—")
        self._do_load_model(model_path, num_classes=len(labels) if labels else 2)
        self.load_data_btn.setEnabled(True)
        if self.dataset_manager is not None:
            self.load_registry_btn.setEnabled(True)

    def load_from_registry(self):
        """Load test signals from the DatasetManager, grouped by modulation."""
        if self.dataset_manager is None:
            return

        entries = self.dataset_manager.scan()
        base_entries = [e for e in entries if not e.get('augmented', False)]
        if not base_entries:
            self.data_label.setText("No datasets in registry")
            return

        # Build label → index map from known class_labels or infer from data
        if self.class_labels:
            label_map = {lbl: i for i, lbl in enumerate(self.class_labels)}
        else:
            mods = sorted({e.get('modulation', e['name']) for e in base_entries})
            label_map = {m: i for i, m in enumerate(mods)}
            self.class_labels = list(label_map.keys())
            self.class_labels_label.setText(", ".join(self.class_labels))

        X_list, y_list = [], []
        for entry in base_entries:
            mod = entry.get('modulation', entry['name'])
            if mod not in label_map:
                continue
            try:
                sig = self.dataset_manager.load_signal(entry)
                X_list.append(np.asarray(sig, dtype=np.float32).ravel())
                y_list.append(label_map[mod])
            except Exception as e:
                print(f"[InferenceTab] Could not load {entry['name']}: {e}")

        if not X_list:
            self.data_label.setText("Failed to load any signals from registry")
            return

        self._build_eval_tensors(X_list, y_list)
        self.data_label.setText(
            f"Loaded {len(X_list)} signals from registry ({len(label_map)} classes)"
        )

    def load_model(self):
        """Load a trained PyTorch model from disk."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", os.path.expanduser("~"),
            "PyTorch Models (*.pth);;All Files (*)"
        )
        if not filepath:
            return

        # Check for a companion JSON sidecar with class labels
        import json as _json
        sidecar = os.path.splitext(filepath)[0] + ".json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar) as f:
                    info = _json.load(f)
                self.class_labels = info.get("class_labels", [])
                self.class_labels_label.setText(
                    ", ".join(self.class_labels) if self.class_labels else "—"
                )
            except Exception:
                pass

        num_classes = len(self.class_labels) if self.class_labels else 2
        self._do_load_model(filepath, num_classes=num_classes)
        if self.dataset_manager is not None:
            self.load_registry_btn.setEnabled(True)

    def _do_load_model(self, filepath: str, num_classes: int = 2):
        """Internal: load model weights and update UI."""
        try:
            from backend.torch_models import get_model
            state_dict = torch.load(filepath, map_location='cpu')
            self.model = get_model('SimpleCNN', num_classes=num_classes)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_path = filepath
            self.model_label.setText(f"Loaded: {os.path.basename(filepath)}")
            self.load_data_btn.setEnabled(True)
        except Exception as e:
            self.model_label.setText(f"Failed to load: {e}")
            print(f"[InferenceTab] Error loading model: {e}")
    
    def load_test_data(self):
        """Load test data from a folder. Each subfolder = one class."""
        folder = QFileDialog.getExistingDirectory(self, "Select Test Data Folder")
        if not folder:
            return

        try:
            subdirs = sorted([
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ])

            X_list, y_list = [], []

            if subdirs:
                # Subfolder-per-class layout
                label_map = (
                    {lbl: i for i, lbl in enumerate(self.class_labels)}
                    if self.class_labels else {}
                )
                inferred = []
                for subdir in subdirs:
                    if subdir in label_map:
                        idx = label_map[subdir]
                    else:
                        idx = len(inferred)
                        inferred.append(subdir)
                    class_path = os.path.join(folder, subdir)
                    for fname in sorted(os.listdir(class_path)):
                        fpath = os.path.join(class_path, fname)
                        if os.path.isfile(fpath) and fname.lower().endswith(('.npy', '.npz', '.csv')):
                            arr = self._load_array(fpath)
                            if arr is not None:
                                X_list.append(np.asarray(arr, dtype=np.float32).ravel())
                                y_list.append(idx)
                if inferred and not self.class_labels:
                    self.class_labels = subdirs
                    self.class_labels_label.setText(", ".join(self.class_labels))
            else:
                # Flat folder — sequential label fallback
                n_cls = max(len(self.class_labels), 2)
                for i, fname in enumerate(sorted(os.listdir(folder))):
                    fpath = os.path.join(folder, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.npy', '.npz', '.csv')):
                        arr = self._load_array(fpath)
                        if arr is not None:
                            X_list.append(np.asarray(arr, dtype=np.float32).ravel())
                            y_list.append(i % n_cls)

            if not X_list:
                self.data_label.setText("No supported files found")
                return

            self._build_eval_tensors(X_list, y_list)
            self.data_label.setText(f"Loaded {len(X_list)} samples from folder")

        except Exception as e:
            self.data_label.setText(f"Error loading data: {e}")
            print(f"[InferenceTab] Error: {e}")

    def _build_eval_tensors(self, X_list: list, y_list: list):
        """Pad and stack signal arrays into eval tensors; enable eval buttons."""
        max_len = max(a.size for a in X_list)
        X = np.zeros((len(X_list), max_len), dtype=np.float32)
        for i, a in enumerate(X_list):
            L = min(len(a), max_len)
            X[i, :L] = a[:L]
        # Shape: (N, 1, L) — standard 1-D CNN input
        X = X[:, np.newaxis, :]
        self.eval_data = torch.from_numpy(X)
        self.eval_labels = np.asarray(y_list, dtype=np.int64)
        self.eval_cm_btn.setEnabled(True)
        self.eval_report_btn.setEnabled(True)
        self.eval_roc_btn.setEnabled(True)
        self.eval_tabs.setEnabled(True)
    
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
            
            self._cm_data = (cm, y_pred)
            self._plot_confusion_matrix(cm, y_pred)
        except Exception as e:
            print(f"[InferenceTab] Error computing confusion matrix: {e}")
    
    def _plot_confusion_matrix(self, cm, y_pred):
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        if self.class_labels and len(self.class_labels) == cm.shape[0]:
            ax.set_xticks(range(cm.shape[1]))
            ax.set_yticks(range(cm.shape[0]))
            ax.set_xticklabels(self.class_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(self.class_labels, fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > thresh else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

        self.cm_figure.colorbar(im, ax=ax)
        _apply_mpl_theme(self.cm_figure, ax)
        self.cm_figure.tight_layout()
        self.cm_canvas.draw()
    
    def evaluate_report(self):
        """Compute and display classification report"""
        if self.model is None or self.eval_data is None:
            return

        try:
            with torch.no_grad():
                outputs = self.model(self.eval_data)
                _, predictions = outputs.max(1)

            y_pred = predictions.cpu().numpy()
            target_names = self.class_labels if self.class_labels else None
            report = classification_report(
                self.eval_labels, y_pred,
                target_names=target_names,
                zero_division=0
            )
            accuracy = (y_pred == self.eval_labels).mean()
            text = f"Accuracy: {accuracy:.4f}\n\n{report}"
            self.report_label.setText(text)
        except Exception as e:
            self.report_label.setText(f"Error: {e}")
            print(f"[InferenceTab] Error: {e}")
    
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
            
            self._roc_data = (fpr, tpr, roc_auc)
            self._plot_roc_curve(fpr, tpr, roc_auc)
        except Exception as e:
            print(f"Error computing ROC curve: {e}")
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc):
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
        
        _apply_mpl_theme(self.roc_figure, ax)
        self.roc_canvas.draw()