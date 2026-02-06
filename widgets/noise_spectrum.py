from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor


class NoiseSpectrumWidget(QWidget):
    """Custom widget to draw noise power spectrum"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 250)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw bars
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(139, 92, 246))
        
        width = self.width()
        height = self.height()
        
        # 6 frequency bands (0-10Hz, 10-20Hz, 20-30Hz, 30-40Hz, 40-50Hz, 50-60Hz)
        bands = [
            (0.05, 0.15, 12),   # 0-10 Hz
            (0.15, 0.30, 25),   # 10-20 Hz
            (0.30, 0.50, 38),   # 20-30 Hz
            (0.50, 0.70, 50),   # 30-40 Hz
            (0.70, 0.85, 30),   # 40-50 Hz
            (0.85, 0.95, 18)    # 50-60 Hz
        ]
        
        for start, end, bar_height_pct in bands:
            x_start = int(width * start)
            x_end = int(width * end)
            bar_width = x_end - x_start
            bar_height = int(height * bar_height_pct / 100)
            painter.drawRect(x_start, height - bar_height, bar_width, bar_height)