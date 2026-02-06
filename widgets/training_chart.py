from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath


class TrainingChartWidget(QWidget):
    """Custom widget to draw training/validation line charts"""
    
    def __init__(self, title="Chart", y_label="Value", parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 250)
        self.title = title
        self.y_label = y_label
        
        # Data storage
        self.training_data = []
        self.validation_data = []
        
        # Chart appearance
        self.margin = 40
        self.training_color = QColor(59, 130, 246)  # Blue
        self.validation_color = QColor(251, 146, 60)  # Orange
        self.grid_color = QColor(240, 240, 240)
        self.axis_color = QColor(200, 200, 200)
        
    def add_data_point(self, training_value, validation_value):
        """Add a new data point to both training and validation series
        
        Args:
            training_value: Training metric value (0.0 to 1.0)
            validation_value: Validation metric value (0.0 to 1.0)
        """
        self.training_data.append(training_value)
        self.validation_data.append(validation_value)
        self.update()  # Trigger repaint
    
    def clear_data(self):
        """Clear all data points"""
        self.training_data = []
        self.validation_data = []
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        width = self.width()
        height = self.height()
        
        # Calculate chart area
        chart_width = width - 2 * self.margin
        chart_height = height - 2 * self.margin
        
        # Draw axes
        self.draw_axes(painter, width, height)
        
        # Draw grid
        self.draw_grid(painter, width, height, chart_height)
        
        # Draw data if available
        if len(self.training_data) > 1:
            self.draw_line(painter, self.training_data, chart_width, chart_height, 
                          self.training_color)
        
        if len(self.validation_data) > 1:
            self.draw_line(painter, self.validation_data, chart_width, chart_height, 
                          self.validation_color)
    
    def draw_axes(self, painter, width, height):
        """Draw X and Y axes"""
        painter.setPen(QPen(self.axis_color, 1))
        
        # Y-axis
        painter.drawLine(self.margin, self.margin, 
                        self.margin, height - self.margin)
        
        # X-axis
        painter.drawLine(self.margin, height - self.margin, 
                        width - self.margin, height - self.margin)
    
    def draw_grid(self, painter, width, height, chart_height):
        """Draw grid lines"""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Horizontal grid lines (5 lines)
        for i in range(1, 5):
            y = self.margin + (chart_height * i / 5)
            painter.drawLine(self.margin, int(y), 
                           width - self.margin, int(y))
    
    def draw_line(self, painter, data, chart_width, chart_height, color):
        """Draw a line for the given data series
        
        Args:
            painter: QPainter object
            data: List of data points (0.0 to 1.0)
            chart_width: Width of chart area
            chart_height: Height of chart area
            color: QColor for the line
        """
        painter.setPen(QPen(color, 2))
        path = QPainterPath()
        
        for i, value in enumerate(data):
            # Clamp value between 0 and 1
            value = max(0.0, min(1.0, value))
            
            # Calculate position
            x = self.margin + (chart_width * i / (len(data) - 1))
            y = self.height() - self.margin - (chart_height * value)
            
            if i == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
        
        painter.drawPath(path)