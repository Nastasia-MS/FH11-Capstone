from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class InferenceResultsTab(QWidget):
    """Inference results visualization tab (placeholder)"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        label = QLabel("Inference Results - Coming Soon")
        label.setAlignment(Qt.AlignCenter)
        label.setProperty("class", "section-title")
        
        layout.addWidget(label)