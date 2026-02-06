import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen


class PowerSpectrumWidget(QWidget):
    """Custom widget to draw power spectrum using sinc² function"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 200)
        self.symbol_rate = 10  # Msps
        self.signal_power = 0  # dBm
        self.bandwidth = 20  # MHz
    
    def set_parameters(self, symbol_rate, signal_power, bandwidth):
        """Update PSD parameters"""
        self.symbol_rate = symbol_rate
        self.signal_power = signal_power
        self.bandwidth = bandwidth
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        width = self.width()
        height = self.height()
        
        # Calculate PSD using sinc^2 function
        num_points = 500
        frequencies = []
        psd_values = []
        
        # Convert signal power from dBm to linear scale
        power_linear = 10 ** (self.signal_power / 10.0)
        
        # Symbol period in microseconds
        T = 1.0 / self.symbol_rate
        
        # Frequency range: show ±bandwidth/2 around center
        f_min = -self.bandwidth / 2.0
        f_max = self.bandwidth / 2.0
        
        for i in range(num_points):
            # Frequency in MHz relative to carrier
            f = f_min + (f_max - f_min) * i / num_points
            
            # Sinc function: sinc(x) = sin(pi*x)/(pi*x)
            x = math.pi * f * T
            if abs(x) < 1e-6:
                sinc_val = 1.0
            else:
                sinc_val = math.sin(x) / x
            
            # PSD = Power * T * sinc^2(pi * f * T)
            psd = power_linear * T * (sinc_val ** 2)
            
            frequencies.append(f)
            psd_values.append(psd)
        
        # Normalize for display
        max_psd = max(psd_values) if psd_values else 1.0
        if max_psd == 0:
            max_psd = 1.0
        
        # Draw spectrum
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(16, 185, 129))
        
        bar_width = width / num_points
        
        for i, psd in enumerate(psd_values):
            # Normalize to height (leave margin at top)
            normalized = psd / max_psd
            bar_height = int(height * 0.9 * normalized)
            
            x = i * bar_width
            painter.drawRect(int(x), height - bar_height, int(bar_width) + 1, bar_height)