import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from tabs.waveform_tab import WaveformSelectionTab
from tabs.channel_tab import ChannelNoiseTab
from tabs.ml_training_tab import MLTrainingTab
from tabs.inference_tab import InferenceResultsTab
from styles.stylesheet import get_stylesheet

from backend.matlab_engine import MatlabEngine
from backend.dataset_manager import DatasetManager

# TODO: waveform_tab and channel_tab appear to share 2 of the same functions

class SignalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Generation & Classification")
        self.setMinimumSize(1400, 900)
        
        # Apply stylesheet
        self.setStyleSheet(get_stylesheet())
        
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
        layout.addLayout(tab_layout)
    
    def switch_tab(self, index):
        """Switch between tabs"""
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)
        self.content_stack.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = SignalDashboard()
    window.show()
    sys.exit(app.exec())