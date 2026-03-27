import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QWizard,
    QWizardPage,
    QFormLayout,
    QLineEdit,
    QFileDialog,
    QComboBox,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont

from tabs.waveform_tab import WaveformSelectionTab
from tabs.channel_tab import ChannelNoiseTab
from tabs.ml_training_tab import MLTrainingTab
from tabs.inference_tab import InferenceResultsTab
from tabs.evaluate_model_tab import EvaluateModelTab
from tabs.data_visualization_tab import DataVisualizationTab
from styles.stylesheet import get_stylesheet, SettingsDialog, apply_theme_palette

from backend.matlab_engine import MatlabEngine
from backend.dataset_manager import DatasetManager



# ----- setup wizard ------------------------------------------------------
class WelcomePage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.setTitle("Welcome")
        text = (
            "<h2>Welcome to the Signal Generation & Classification GUI</h2>"
            "<p>This short tour will show you the main sections of the application:</p>"
            "<ul>"
            "<li><b>Waveform Selection</b> – configure and generate waveforms.</li>"
            "<li><b>Channel & Noise</b> – apply channel models to signals.</li>"
            "<li><b>ML Training</b> – train classifiers on your datasets.</li>"
            "<li><b>Inference Results</b> – inspect prediction performance.</li>"
            "<li><b>Evaluate Model</b> – test models with generated waveforms.</li>"
            "<li><b>Data Visualization</b> – explore datasets with PCA/t-SNE/UMAP.</li>"
            "</ul>"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class SettingsPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.setTitle("Configuration")
        self.setSubTitle("Choose where to save/load models and data, and select compute mode.")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight)

        self.model_path = QLineEdit()
        self.model_path.setMinimumWidth(350)
        self.model_path.setPlaceholderText("Select model folder…")
        btn_model = QPushButton("Browse…")
        btn_model.setFixedWidth(80)
        btn_model.clicked.connect(lambda: self._choose(self.model_path))
        hl_model = QHBoxLayout()
        hl_model.addWidget(self.model_path, 1)
        hl_model.addWidget(btn_model)
        form.addRow("Model folder:", hl_model)
        self.registerField("modelPath*", self.model_path)

        self.data_path = QLineEdit()
        self.data_path.setMinimumWidth(350)
        self.data_path.setPlaceholderText("Select data folder…")
        btn_data = QPushButton("Browse…")
        btn_data.setFixedWidth(80)
        btn_data.clicked.connect(lambda: self._choose(self.data_path))
        hl_data = QHBoxLayout()
        hl_data.addWidget(self.data_path, 1)
        hl_data.addWidget(btn_data)
        form.addRow("Data folder:", hl_data)
        self.registerField("dataPath*", self.data_path)

        self.gpu_label = QLabel()
        form.addRow("Detected GPUs:", self.gpu_label)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["CPU", "GPU"])
        form.addRow("Compute mode:", self.mode_box)
        self.registerField("computeMode", self.mode_box, "currentText", self.mode_box.currentTextChanged)

        layout.addLayout(form)
        layout.addStretch()

    def _choose(self, widget: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            widget.setText(path)


class WaveformPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("Waveform Selection")
        self.dashboard = dashboard
        text = (
            "<b>Waveform Selection Tab</b><br><br>"
            "Here you can:<br>"
            "• Choose modulation types (PAM, QAM, PSK, FSK, FHSS)<br>"
            "• Configure sampling frequency, carrier frequency, symbol period<br>"
            "• Adjust modulation order, noise variance, and pulse shape<br>"
            "• Generate single or batch waveforms<br>"
            "• Preview the signal in time, frequency, and constellation domains"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class ChannelPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("Channel & Noise")
        self.dashboard = dashboard
        text = (
            "<b>Channel & Noise Tab</b><br><br>"
            "Here you can:<br>"
            "• Apply channel models (AWGN, fading, etc.)<br>"
            "• Add signal degradation and impairments<br>"
            "• Visualize the affected waveforms<br>"
            "• Save processed datasets for training"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class MLTrainingPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("ML Training")
        self.dashboard = dashboard
        text = (
            "<b>ML Training Tab</b><br><br>"
            "Here you can:<br>"
            "• Load datasets generated in the Waveform & Channel tabs<br>"
            "• Train deep learning models (PyTorch or TensorFlow)<br>"
            "• Configure training parameters (epochs, batch size, etc.)<br>"
            "• Monitor training progress and validation accuracy<br>"
            "• Save trained models for later inference"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class InferencePage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("Inference Results")
        self.dashboard = dashboard
        text = (
            "<b>Inference Results Tab</b><br><br>"
            "Here you can:<br>"
            "• Load trained models and test datasets<br>"
            "• Run predictions on unknown signals<br>"
            "• Visualize confusion matrices<br>"
            "• Inspect accuracy, precision, recall metrics<br>"
            "• Export classification reports"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class EvaluateModelPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("Evaluate Model")
        self.dashboard = dashboard
        text = (
            "<b>Evaluate Model Tab</b><br><br>"
            "Here you can:<br>"
            "• Load a trained model (.pth file)<br>"
            "• Generate custom waveforms with specific parameters<br>"
            "• Classify generated signals in real-time<br>"
            "• View classification confidence scores<br>"
            "• Visualize waveforms in time, frequency, and constellation domains"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class DataVisualizationPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setTitle("Data Visualization")
        self.dashboard = dashboard
        text = (
            "<b>Data Visualization Tab</b><br><br>"
            "Here you can:<br>"
            "• Load datasets from the waveform_data folder or datasets registry<br>"
            "• Apply dimensionality reduction (PCA, t-SNE, UMAP)<br>"
            "• Visualize signal clusters in 2D and 3D scatter plots<br>"
            "• Configure feature extraction (raw samples, FFT, statistics)<br>"
            "• Filter classes and interactively explore data distributions"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)


class TipsPage(QWizardPage):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.setTitle("Tips & Tricks")
        label = QLabel(
            "Use the tab buttons at the top to switch between workflows. "
            "Click 'Settings ⚙' in the top bar to change the colour theme, font, and other preferences."
        )
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

class SetupWizard(QWizard):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup & Quick Tour")
        self.setMinimumWidth(600)
        self.dashboard = dashboard

        self.page_welcome   = WelcomePage(dashboard=dashboard)
        self.page_waveform  = WaveformPage(dashboard=dashboard)
        self.page_channel   = ChannelPage(dashboard=dashboard)
        self.page_ml        = MLTrainingPage(dashboard=dashboard)
        self.page_inference = InferencePage(dashboard=dashboard)
        self.page_evaluate  = EvaluateModelPage(dashboard=dashboard)
        self.page_dataviz   = DataVisualizationPage(dashboard=dashboard)
        self.page_settings  = SettingsPage(dashboard=dashboard)
        self.page_tips      = TipsPage(dashboard=dashboard)

        self.id_welcome   = self.addPage(self.page_welcome)
        self.id_waveform  = self.addPage(self.page_waveform)
        self.id_channel   = self.addPage(self.page_channel)
        self.id_ml        = self.addPage(self.page_ml)
        self.id_inference = self.addPage(self.page_inference)
        self.id_evaluate  = self.addPage(self.page_evaluate)
        self.id_dataviz   = self.addPage(self.page_dataviz)
        self.id_settings  = self.addPage(self.page_settings)
        self.id_tips      = self.addPage(self.page_tips)

        self.page_settings.gpu_label.setText(str(detect_gpus()))
        if int(self.page_settings.gpu_label.text()) == 0:
            self.page_settings.mode_box.setCurrentText("CPU")
            self.page_settings.mode_box.setEnabled(False)

        # Map page id → (dashboard tab index or None)
        # This fires for both Next AND Back, solving the back-button bug.
        self._tab_map = {
            self.id_welcome:   None,
            self.id_waveform:  0,
            self.id_channel:   1,
            self.id_ml:        2,
            self.id_inference: 3,
            self.id_evaluate:  4,
            self.id_dataviz:   5,
            self.id_settings:  None,
            self.id_tips:      None,
        }
        self.currentIdChanged.connect(self._sync_tab)

    def _sync_tab(self, page_id):
        """Highlight the matching dashboard tab (or clear all) whenever the page changes."""
        if not self.dashboard:
            return
        tab_index = self._tab_map.get(page_id)
        for btn in self.dashboard.tab_buttons:
            btn.setStyleSheet("")
        if tab_index is not None:
            self.dashboard.switch_tab(tab_index)
            self.dashboard.tab_buttons[tab_index].setStyleSheet(
                "QPushButton { background-color: #FFD700; font-weight: bold; }"
            )

    def results(self):
        return {
            "modelPath": self.field("modelPath"),
            "dataPath": self.field("dataPath"),
            "mode": self.field("computeMode"),
        }
    

# -------------------------------------------------------------------------


def detect_gpus():
    """Return number of GPUs available according to torch or tensorflow."""
    count = 0
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        try:
            import tensorflow as tf
            count = len(tf.config.list_physical_devices("GPU"))
        except Exception:
            count = 0
    return count


# TODO: waveform_tab and channel_tab appear to share 2 of the same functions

class SignalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Signal Generation & Classification")
        self.setMinimumSize(1400, 900)
        
        self.setStyleSheet(get_stylesheet())
        apply_theme_palette()
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Tab navigation
        self.create_tab_navigation(main_layout)

        # Menu bar
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")

        settings_act = help_menu.addAction("Settings…")
        settings_act.triggered.connect(self.show_settings_dialog)

        help_menu.addSeparator()

        show_wizard_act = help_menu.addAction("Show welcome tutorial")
        show_wizard_act.triggered.connect(self.show_init_dialog)
        reset_act = help_menu.addAction("Reset welcome settings")
        reset_act.triggered.connect(self._reset_wizard)
        
        # Content area (stacked widget for different tabs)
        self.content_stack = QStackedWidget()

        self.matlab = MatlabEngine(lazy=True)

        self.matlab.start()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        waveform_path = os.path.join(script_dir, 'waveform_functions')
        print(f"Adding MATLAB path: {waveform_path}")

        if self.matlab.is_available():
            self.matlab.add_path(waveform_path)
            print(f"MATLAB engine ready. Waveform path added: {waveform_path}")
        else:
            print("MATLAB engine unavailable — waveform generation will be disabled. Install MATLAB and the MATLAB Engine for Python, or call MatlabEngine.start() to try starting it.")
        
        # Initialize Dataset Manager (shared across tabs)
        print("Initializing Dataset Manager...")
        settings = QSettings("MyCompany", "MixedSignalGUI")
        data_folder = settings.value("dataPath", "")
        if data_folder:
            print(f"Using data folder from settings: {data_folder}")
            self.dataset_manager = DatasetManager(datasets_dir=data_folder)
        else:
            print("No data folder set in settings, using default location")
            self.dataset_manager = DatasetManager()
        
        # Add all tabs
        self.waveform_tab = WaveformSelectionTab(self.matlab, self.dataset_manager)
        self.content_stack.addWidget(self.waveform_tab)
        
        self.channel_tab = ChannelNoiseTab(self.dataset_manager)
        self.content_stack.addWidget(self.channel_tab)
        
        self.ml_training_tab = MLTrainingTab(dataset_manager=self.dataset_manager)
        self.content_stack.addWidget(self.ml_training_tab)
        
        self.inference_tab = InferenceResultsTab(dataset_manager=self.dataset_manager)
        self.content_stack.addWidget(self.inference_tab)

        self.evaluate_tab = EvaluateModelTab(self.matlab, dataset_manager=self.dataset_manager)
        self.content_stack.addWidget(self.evaluate_tab)

        self.data_viz_tab = DataVisualizationTab(dataset_manager=self.dataset_manager)
        self.content_stack.addWidget(self.data_viz_tab)
        
        main_layout.addWidget(self.content_stack, 1)

        # Wire ML tab → Inference + Evaluate tabs: auto-load model when training completes
        self.ml_training_tab.trained_model_ready.connect(
            self.inference_tab.on_trained_model_ready
        )
        self.ml_training_tab.trained_model_ready.connect(
            self.evaluate_tab.receive_trained_model
        )
    
    def create_header(self, layout):
        """Create the header section with title and subtitle"""
        header_layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        title = QLabel("📈 Signal Generation & Classification")
        title.setProperty("class", "title")
        subtitle = QLabel("Configure waveforms, channels, train ML models, and analyze classification results")
        subtitle.setProperty("class", "subtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(4)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
    
    def create_tab_navigation(self, layout):
        """Create the tab navigation buttons"""
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(0)
        
        self.tab_buttons = []
        tabs = ["Waveform Selection", "Channel & Noise", "ML Training", "Inference Results", "Evaluate Model", "Data Visualization"]
        
        for i, tab_name in enumerate(tabs):
            tab_btn = QPushButton(tab_name)
            tab_btn.setObjectName("tabButton")
            tab_btn.setCheckable(True)
            if i == 0:
                tab_btn.setChecked(True)
            tab_btn.clicked.connect(lambda checked, idx=i: self.switch_tab(idx))
            self.tab_buttons.append(tab_btn)
            tab_layout.addWidget(tab_btn)
        
        tab_layout.addStretch()
        
        # Settings button (replaces old Help button)
        settings_btn = QPushButton("Settings ⚙")
        settings_btn.setMaximumWidth(120)
        settings_btn.clicked.connect(self.show_settings_dialog)
        tab_layout.addWidget(settings_btn)
        
        layout.addLayout(tab_layout)
    
    def switch_tab(self, index):
        """Switch between tabs"""
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
        self.content_stack.setCurrentIndex(index)

    def apply_theme(self):
        self.setStyleSheet(get_stylesheet())
        apply_theme_palette()
        
        s = QSettings("MyCompany", "MixedSignalGUI")
        font_family = s.value("fontFamily", "Segoe UI")
        font_size   = int(s.value("fontSize", 10))
        QApplication.instance().setFont(QFont(font_family, font_size))
        
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._refresh_all_plots)
    
    def _refresh_all_plots(self):
        tabs_with_plots = [
            self.waveform_tab,
            self.channel_tab,
            self.data_viz_tab,
            self.evaluate_tab,
        ]
        
        for tab in tabs_with_plots:
            self._trigger_replots_in_widget(tab)
    
    def _trigger_replots_in_widget(self, widget):
        from widgets.waveform_plots import PlottingWidget
        from widgets.comparison_widget import ComparisonWidget
        from PySide6.QtCore import QTimer
        
        plot_widgets = []
        
        if isinstance(widget, (PlottingWidget, ComparisonWidget)):
            plot_widgets.append(widget)
        
        plot_widgets.extend(widget.findChildren(PlottingWidget))
        plot_widgets.extend(widget.findChildren(ComparisonWidget))
        
        for plot_widget in plot_widgets:
            QTimer.singleShot(0, plot_widget._replot_with_theme)

    def show_settings_dialog(self):
        # Store old data path to detect changes
        settings = QSettings("MyCompany", "MixedSignalGUI")
        old_data_path = settings.value("dataPath", "")
        
        dlg = SettingsDialog(parent=self)
        dlg.settingsChanged.connect(self.apply_theme)
        dlg.exec()
        self.apply_theme()
        
        # Check if data path changed
        new_data_path = settings.value("dataPath", "")
        if new_data_path != old_data_path:
            print(f"Data path changed from '{old_data_path}' to '{new_data_path}'")
            print("Reinitializing Dataset Manager...")
            if new_data_path:
                self.dataset_manager = DatasetManager(datasets_dir=new_data_path)
            else:
                self.dataset_manager = DatasetManager()
            
            # Update all tabs with new dataset_manager
            self.waveform_tab.dataset_manager = self.dataset_manager
            self.channel_tab.dataset_manager = self.dataset_manager
            self.ml_training_tab.dataset_manager = self.dataset_manager
            self.inference_tab.dataset_manager = self.dataset_manager
            self.evaluate_tab.dataset_manager = self.dataset_manager
            self.data_viz_tab.dataset_manager = self.dataset_manager
            print("All tabs updated with new dataset manager")

    def show_init_dialog(self):
        """Check if first run and show wizard if needed. Called at startup."""
        settings = QSettings("MyCompany", "MixedSignalGUI")
        initialized = settings.value("initialized", False, type=bool)
        if initialized:
            return
        self.show_wizard()

    def show_wizard(self):
        wiz = SetupWizard(dashboard=self, parent=self)
        if wiz.exec() == QWizard.Accepted:
            res = wiz.results()
            settings = QSettings("MyCompany", "MixedSignalGUI")
            settings.setValue("modelPath", res.get("modelPath", ""))
            settings.setValue("dataPath", res.get("dataPath", ""))
            settings.setValue("mode", res.get("mode", "CPU"))
            settings.setValue("initialized", True)
            
            # Reinitialize dataset_manager with new data path
            data_path = res.get("dataPath", "")
            if data_path:
                print(f"Wizard complete. Using data folder: {data_path}")
                self.dataset_manager = DatasetManager(datasets_dir=data_path)
            else:
                print("Wizard complete. Using default data location")
                self.dataset_manager = DatasetManager()
            
            # Update all tabs with new dataset_manager
            self.waveform_tab.dataset_manager = self.dataset_manager
            self.channel_tab.dataset_manager = self.dataset_manager
            self.ml_training_tab.dataset_manager = self.dataset_manager
            self.inference_tab.dataset_manager = self.dataset_manager
            self.evaluate_tab.dataset_manager = self.dataset_manager
            self.data_viz_tab.dataset_manager = self.dataset_manager
        
        # clear all tab highlights when wizard closes
        for btn in self.tab_buttons:
            btn.setStyleSheet("")
            btn.setStyleSheet("")

    def _reset_wizard(self):
        settings = QSettings("MyCompany", "MixedSignalGUI")
        settings.remove("initialized")
        self.show_init_dialog()

    def model_dir(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("modelPath", "")

    def data_dir(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("dataPath", "")

    def compute_mode(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("mode", "CPU")


def main():
    app = QApplication(sys.argv)
    
    # Allow --reset-wizard flag to force wizard on next launch
    if "--reset-wizard" in sys.argv:
        settings = QSettings("MyCompany", "MixedSignalGUI")
        settings.remove("initialized")
        print(f"✓ Wizard flag cleared (file: {settings.fileName()})")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = SignalDashboard()
    window.show()

    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, window.show_init_dialog)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()