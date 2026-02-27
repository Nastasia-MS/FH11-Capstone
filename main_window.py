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
    QTextEdit,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont

from tabs.waveform_tab import WaveformSelectionTab
from tabs.channel_tab import ChannelNoiseTab
from tabs.ml_training_tab import MLTrainingTab
from tabs.inference_tab import InferenceResultsTab
from styles.stylesheet import get_stylesheet

from backend.matlab_engine import MatlabEngine
from backend.dataset_manager import DatasetManager



# ----- setup wizard ------------------------------------------------------
class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome")
        text = (
            "<h2>Welcome to the Signal Generation & Classification GUI</h2>"
            "<p>This short tour will show you the main sections of the application:</p>"
            "<ul>"
            "<li><b>Waveform Selection</b> – configure and generate waveforms.</li>"
            "<li><b>Channel & Noise</b> – apply channel models to signals.</li>"
            "<li><b>ML Training</b> – train classifiers on your datasets.</li>"
            "<li><b>Inference Results</b> – inspect prediction performance.</li>"
            "</ul>"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

class SettingsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Configuration")
        self.setSubTitle("Choose where to save/load models and data, and select compute mode.")

        layout = QFormLayout(self)

        self.model_path = QLineEdit()
        btn_model = QPushButton("…")
        btn_model.clicked.connect(lambda: self._choose(self.model_path))
        hl_model = QHBoxLayout()
        hl_model.addWidget(self.model_path)
        hl_model.addWidget(btn_model)
        layout.addRow("Model folder:", hl_model)
        self.registerField("modelPath*", self.model_path)

        self.data_path = QLineEdit()
        btn_data = QPushButton("…")
        btn_data.clicked.connect(lambda: self._choose(self.data_path))
        hl_data = QHBoxLayout()
        hl_data.addWidget(self.data_path)
        hl_data.addWidget(btn_data)
        layout.addRow("Data folder:", hl_data)
        self.registerField("dataPath*", self.data_path)

        self.gpu_label = QLabel()
        layout.addRow("Detected GPUs:", self.gpu_label)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["CPU", "GPU"])
        layout.addRow("Compute mode:", self.mode_box)
        self.registerField("computeMode", self.mode_box, "currentText", self.mode_box.currentTextChanged)

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

    def initializePage(self):
        if self.dashboard:
            for btn in self.dashboard.tab_buttons:
                btn.setStyleSheet("")
            self.dashboard.switch_tab(0)
            self.dashboard.tab_buttons[0].setStyleSheet(
                "QPushButton { background-color: #FFD700; font-weight: bold; }"
            )

    def cleanupPage(self):
        if self.dashboard:
            self.dashboard.tab_buttons[0].setStyleSheet("")

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

    def initializePage(self):
        if self.dashboard:
            for btn in self.dashboard.tab_buttons:
                btn.setStyleSheet("")
            self.dashboard.switch_tab(1)
            self.dashboard.tab_buttons[1].setStyleSheet(
                "QPushButton { background-color: #FFD700; font-weight: bold; }"
            )

    def cleanupPage(self):
        if self.dashboard:
            self.dashboard.tab_buttons[1].setStyleSheet("")

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

    def initializePage(self):
        if self.dashboard:
            for btn in self.dashboard.tab_buttons:
                btn.setStyleSheet("")
            self.dashboard.switch_tab(2)
            self.dashboard.tab_buttons[2].setStyleSheet(
                "QPushButton { background-color: #FFD700; font-weight: bold; }"
            )

    def cleanupPage(self):
        if self.dashboard:
            self.dashboard.tab_buttons[2].setStyleSheet("")

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

    def initializePage(self):
        if self.dashboard:
            for btn in self.dashboard.tab_buttons:
                btn.setStyleSheet("")
            self.dashboard.switch_tab(3)
            self.dashboard.tab_buttons[3].setStyleSheet(
                "QPushButton { background-color: #FFD700; font-weight: bold; }"
            )

    def cleanupPage(self):
        if self.dashboard:
            self.dashboard.tab_buttons[3].setStyleSheet("")

class TipsPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Tips & Tricks")
        label = QLabel(
            "Use the tab buttons at the top to switch between workflows. "
            "You can reopen this tutorial from Help → Show welcome tutorial anytime."
        )
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

class SetupWizard(QWizard):
    def __init__(self, dashboard=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup & Quick Tour")
        self.dashboard = dashboard
        
        self.addPage(WelcomePage())
        self.addPage(WaveformPage(dashboard=dashboard))
        self.addPage(ChannelPage(dashboard=dashboard))
        self.addPage(MLTrainingPage(dashboard=dashboard))
        self.addPage(InferencePage(dashboard=dashboard))
        
        settings_page = SettingsPage()
        settings_page.gpu_label.setText(str(detect_gpus()))
        if int(settings_page.gpu_label.text()) == 0:
            settings_page.mode_box.setCurrentText("CPU")
            settings_page.mode_box.setEnabled(False)
        self.addPage(settings_page)
        
        self.addPage(TipsPage())

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
        
        # Apply stylesheet
        #self.setStyleSheet(get_stylesheet())
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        #self.create_header(main_layout)
        
        # Tab navigation
        self.create_tab_navigation(main_layout)

        # Help menu (for resetting/showing the wizard manually)
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        show_wizard_act = help_menu.addAction("Show welcome tutorial")
        show_wizard_act.triggered.connect(self.show_init_dialog)
        reset_act = help_menu.addAction("Reset welcome settings")
        reset_act.triggered.connect(self._reset_wizard)
        print(f"[debug] Help menu created: {help_menu}")
        print(f"[debug] menubar: {menubar}")
        
        # Content area (stacked widget for different tabs)
        self.content_stack = QStackedWidget()

        self.matlab = MatlabEngine(lazy=True)

        self.matlab.start()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        waveform_path = os.path.join(script_dir, 'waveform_functions')
        print(f"Adding MATLAB path: {waveform_path}")

        if self.matlab.is_available():
            self.matlab.add_path(waveform_path)
            try:
                result = self.matlab.eng.which('plotspec_gui')
                print(f"Found waveform_generator at: {result}")
            except Exception:
                print("ERROR: waveform_generator not found in MATLAB path!")
        else:
            print("MATLAB engine unavailable — waveform generation will be disabled. Install MATLAB and the MATLAB Engine for Python, or call MatlabEngine.start() to try starting it.")
        
        # Initialize Dataset Manager (shared across tabs)
        print("Initializing Dataset Manager...")
        self.dataset_manager = DatasetManager()
        
        # Add all tabs
        self.waveform_tab = WaveformSelectionTab(self.matlab, self.dataset_manager)
        self.content_stack.addWidget(self.waveform_tab)
        
        self.channel_tab = ChannelNoiseTab(self.dataset_manager)
        self.content_stack.addWidget(self.channel_tab)
        
        self.ml_training_tab = MLTrainingTab()
        self.content_stack.addWidget(self.ml_training_tab)
        
        self.inference_tab = InferenceResultsTab()
        self.content_stack.addWidget(self.inference_tab)
        
        main_layout.addWidget(self.content_stack, 1)
    
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
        tabs = ["Waveform Selection", "Channel & Noise", "ML Training", "Inference Results"]
        
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
        
        # Add a help button on the right
        help_btn = QPushButton("? Help")
        help_btn.setMaximumWidth(100)
        help_btn.clicked.connect(self.show_init_dialog)
        tab_layout.addWidget(help_btn)
        
        layout.addLayout(tab_layout)
    
    def switch_tab(self, index):
        """Switch between tabs"""
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
        self.content_stack.setCurrentIndex(index)

    def show_init_dialog(self):
        """Check if first run and show wizard if needed. Called at startup."""
        settings = QSettings("MyCompany", "MixedSignalGUI")
        initialized = settings.value("initialized", False, type=bool)
        print(f"[debug] show_init_dialog called, initialized={initialized}")
        print(f"[debug] settings file: {settings.fileName()}")
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
        
        # clear all tab highlights when wizard closes
        for btn in self.tab_buttons:
            btn.setStyleSheet("")

    def _reset_wizard(self):
        settings = QSettings("MyCompany", "MixedSignalGUI")
        settings.remove("initialized")
        print("[debug] wizard initialization flag cleared")
        # optionally show it again
        self.show_init_dialog()

    def model_dir(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("modelPath", "")

    def data_dir(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("dataPath", "")

    def compute_mode(self):
        return QSettings("MyCompany", "MixedSignalGUI").value("mode", "CPU")


if __name__ == "__main__":
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