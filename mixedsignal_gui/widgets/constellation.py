import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen


class ConstellationWidget(QWidget):
    """Custom widget to draw constellation diagram"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 220)
        self.setMaximumSize(600, 220)
        self.modulation_type = "QPSK"
        self.update_points()
    
    def set_modulation(self, mod_type):
        """Update constellation points based on modulation type"""
        self.modulation_type = mod_type
        self.update_points()
        self.update()
    
    def update_points(self):
        """Calculate constellation points based on modulation"""
        if self.modulation_type == "BPSK":
            self.points = [(-1, 0), (1, 0)]
        elif self.modulation_type == "QPSK":
            self.points = [(0.75, 0.75), (-0.75, 0.75), (-0.75, -0.75), (0.75, -0.75)]
        elif self.modulation_type == "8PSK":
            self.points = []
            for i in range(8):
                angle = 2 * math.pi * i / 8 + math.pi / 8
                self.points.append((0.75 * math.cos(angle), 0.75 * math.sin(angle)))
        elif self.modulation_type == "16QAM":
            self.points = []
            for i in [-0.75, -0.25, 0.25, 0.75]:
                for j in [-0.75, -0.25, 0.25, 0.75]:
                    self.points.append((i, j))
        elif self.modulation_type == "64QAM":
            self.points = []
            vals = [-0.857, -0.612, -0.367, -0.122, 0.122, 0.367, 0.612, 0.857]
            for i in vals:
                for j in vals:
                    self.points.append((i, j))
        else:
            self.points = [(0.75, 0.75), (-0.75, 0.75), (-0.75, -0.75), (0.75, -0.75)]
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw axes
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        mid_x = self.width() // 2
        mid_y = self.height() // 2
        painter.drawLine(0, mid_y, self.width(), mid_y)
        painter.drawLine(mid_x, 0, mid_x, self.height())
        
        # Draw constellation points
        painter.setPen(QPen(QColor(59, 130, 246), 8))
        scale = min(self.width(), self.height()) * 0.35
        for x, y in self.points:
            px = mid_x + int(x * scale)
            py = mid_y - int(y * scale)
            painter.drawPoint(px, py)