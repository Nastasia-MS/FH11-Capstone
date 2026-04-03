from PySide6.QtWidgets import QCheckBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor


class ToggleSwitch(QCheckBox):
    """Custom toggle switch widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 24)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background track
        if self.isChecked():
            painter.setBrush(QColor(17, 24, 39))  # Dark when on
        else:
            painter.setBrush(QColor(209, 213, 219))  # Gray when off
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 44, 24, 12, 12)
        
        # Slider circle
        painter.setBrush(QColor(255, 255, 255))
        if self.isChecked():
            painter.drawEllipse(22, 2, 20, 20)
        else:
            painter.drawEllipse(2, 2, 20, 20)