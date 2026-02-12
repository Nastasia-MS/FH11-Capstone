from PySide6.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy import signal

from widgets.signal_utils import demodulate_to_symbols, extract_fsk_iq, extract_fhss_iq

class PlottingWidget(QWidget):
    def __init__(self):
        super().__init__()
                
        layout = QVBoxLayout()
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar for interactive controls (zoom, pan, etc.)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add toolbar and canvas to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create a button to update/refresh the plot
        self.refresh_button = QPushButton("Refresh Plot")
        self.refresh_button.clicked.connect(self.plot_data)
        layout.addWidget(self.refresh_button)
        
        self.setLayout(layout)
        
        # Initial plot
        self.plot_data()
    
    def plot_data(self, t=None, signal=None, signal_q=None):
        """Generate and display a sample plot. If signal_q is provided, plots I and Q traces."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if t is not None and signal is not None:
            if signal_q is not None:
                ax.plot(t, signal, label='I (In-phase)', alpha=0.8)
                ax.plot(t, signal_q, label='Q (Quadrature)', alpha=0.8)
            else:
                ax.plot(t, signal, label='Waveform')
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Waveform Plot')
            ax.legend()
            ax.grid(True)

        self.canvas.draw()


class FreqDomainPlot(PlottingWidget):
    def __init__(self):
        super().__init__()
        
    def plot_data(self, freqs=None, fft=None):
        """Generate and display a sample plot"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
     
        if (freqs is not None and fft is not None):
            ax.plot(freqs, fft, label='Waveform')
            ax.set_xlabel('Frequency [MHz]')
            ax.set_ylabel('Magnitude')
            ax.set_title('Frequency Domain')
            ax.legend()
            ax.grid(True)
        
        self.canvas.draw()


class IQDomainPlot(PlottingWidget):
    def __init__(self):
        super().__init__()
        self.refresh_button.hide()
        
    def plot_data(self, data=None, fs=None, fc=None, sps=None, M=None,
                  modulation=None, alpha=0.35, span=8, pulse_shape='rrc',
                  nsymb=None):
        """Plot IQ constellation using matched filter demodulation."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if data is None or modulation is None:
            ax.text(0.5, 0.5, 'No IQ data to display', 
                   ha='center', va='center', fontsize=14, color='gray')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            self.canvas.draw()
            return
        
        if modulation == "FSK":
            self._plot_fsk_trajectory(ax, data, fs, fc, sps, M)
        elif modulation == "FHSS":
            self._plot_fhss_trajectory(ax, data, fs, fc, sps, M)
        else:
            self._plot_constellation(ax, data, fs, fc, sps, M, modulation,
                                     alpha, span, pulse_shape, nsymb)
        
        self.canvas.draw()
    
    def _plot_constellation(self, ax, data, fs, fc, sps, M, modulation,
                            alpha, span, pulse_shape, nsymb):
        """Plot constellation using matched filter demodulation."""
        M = int(M)

        rxRecovered = demodulate_to_symbols(
            data, fs, fc, int(sps), alpha=alpha, span=int(span),
            pulse_shape=pulse_shape, nsymb=nsymb,
        )

        I_symbols = np.real(rxRecovered)
        Q_symbols = np.imag(rxRecovered)

        ax.scatter(I_symbols, Q_symbols, alpha=0.6, s=20, label='Received Symbols')
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_title(f"{M}-{modulation} Constellation Diagram")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        
        if modulation == "QAM":
            self._add_ideal_qam_points(ax, M)
        elif modulation == "PSK":
            self._add_ideal_psk_points(ax, M)
        elif modulation == "PAM":
            self._add_ideal_pam_points(ax, M)
            ax.text(0.98, 0.02, 'PAM: Q ≈ 0 (amplitude modulation only)', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_fsk_trajectory(self, ax, data, fs, fc, sps, M):
        """Plot FSK frequency trajectory in IQ space"""
        I_viz, Q_viz = extract_fsk_iq(data, fs, fc, sps, M)
        
        time_colors = np.arange(len(I_viz))
        
        scatter = ax.scatter(I_viz, Q_viz, c=time_colors, 
                            cmap='viridis', alpha=0.5, s=10)
        
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_title(f"{int(M)}-FSK IQ Trajectory (colored by time)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        
        self.figure.colorbar(scatter, ax=ax, label='Time')
    
    def _plot_fhss_trajectory(self, ax, data, fs, fc, sps, M):
        """Plot FHSS frequency hopping trajectory in IQ space"""
        I_viz, Q_viz = extract_fhss_iq(data, fs, fc, sps, M)
        
        time_colors = np.arange(len(I_viz))
        
        scatter = ax.scatter(I_viz, Q_viz, c=time_colors, 
                            cmap='plasma', alpha=0.5, s=10)
        
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_title(f"FHSS IQ Trajectory ({int(M)} channels, colored by time)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        
        self.figure.colorbar(scatter, ax=ax, label='Time')
    
    def _add_ideal_qam_points(self, ax, M):
        """Add ideal QAM constellation points"""
        M = int(M)
        sqrt_M = int(np.sqrt(M))
        
        if sqrt_M * sqrt_M != M:
            return
        
        levels = np.arange(-(sqrt_M-1), sqrt_M, 2)
        I_ideal, Q_ideal = np.meshgrid(levels, levels)
        I_ideal = I_ideal.flatten()
        Q_ideal = Q_ideal.flatten()
        
        norm_factor = np.sqrt(np.mean(I_ideal**2 + Q_ideal**2))
        I_ideal = I_ideal / norm_factor
        Q_ideal = Q_ideal / norm_factor
        
        ax.scatter(I_ideal, Q_ideal, c='red', marker='x', s=100, 
                  linewidths=2, label='Ideal', zorder=5)
        ax.legend()
    
    def _add_ideal_psk_points(self, ax, M):
        """Add ideal PSK points on unit circle"""
        M = int(M)
        angles = 2 * np.pi * np.arange(M) / M
        I_ideal = np.cos(angles)
        Q_ideal = np.sin(angles)
        
        ax.scatter(I_ideal, Q_ideal, c='red', marker='x', s=100, 
                  linewidths=2, label='Ideal', zorder=5)
        
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'g--', alpha=0.3, label='Unit Circle')
        ax.legend()
    
    def _add_ideal_pam_points(self, ax, M):
        """Add ideal PAM levels on I-axis"""
        M = int(M)
        levels = np.arange(-(M-1), M, 2).astype(float)
        
        norm_factor = np.sqrt(np.mean(levels**2))
        if norm_factor > 0:
            levels = levels / norm_factor
        
        ax.scatter(levels, np.zeros_like(levels), c='red', marker='x', 
                  s=100, linewidths=2, label='Ideal PAM Levels', zorder=5)
        
        for level in levels:
            ax.axvline(x=level, color='red', linewidth=0.5, 
                      alpha=0.2, linestyle='--')
        
        ax.legend()


class SpectrogramPlot(PlottingWidget):
    def __init__(self):
        super().__init__()
        
        self.refresh_button.hide()
        self._create_controls()
        
        self.current_x = None
        self.current_fs = None
        self.current_modulation = None
    
    def _create_controls(self):
        """Create minimal control panel"""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        
        cmap_label = QLabel("Color Scheme:")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "jet", "hot"])
        
        self.update_button = QPushButton("Refresh")
        self.update_button.clicked.connect(self._update_plot)
        
        controls_layout.addWidget(cmap_label)
        controls_layout.addWidget(self.cmap_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.update_button)
        
        controls_widget.setLayout(controls_layout)
        self.layout().insertWidget(0, controls_widget)
    
    def plot_data(self, x=None, fs=None, modulation=None):
        if x is not None and fs is not None:
            self.current_x = x
            self.current_fs = fs
            self.current_modulation = modulation
            self._update_plot()
    
    def _get_preset(self):
        default = {"nperseg": 1024, "overlap": 0.75, "window": "hann", 
                   "vmin_pct": 5, "vmax_pct": 95}
        
        presets = {
            "PAM": {"nperseg": 512, "overlap": 0.70, "window": "hann", 
                    "vmin_pct": 10, "vmax_pct": 95},
            "QAM": {"nperseg": 1024, "overlap": 0.75, "window": "hann", 
                    "vmin_pct": 5, "vmax_pct": 95},
            "PSK": {"nperseg": 1024, "overlap": 0.75, "window": "hann", 
                    "vmin_pct": 5, "vmax_pct": 95},
            "ASK": {"nperseg": 512, "overlap": 0.70, "window": "hann", 
                    "vmin_pct": 10, "vmax_pct": 95},
            "FSK": {"nperseg": 2048, "overlap": 0.85, "window": "blackman", 
                    "vmin_pct": 3, "vmax_pct": 97},
            "OFDM": {"nperseg": 2048, "overlap": 0.80, "window": "hann", 
                     "vmin_pct": 5, "vmax_pct": 90}
        }
        
        return presets.get(self.current_modulation, default)
    
    def _update_plot(self):
        if self.current_x is None or self.current_fs is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        x = self.current_x
        fs = self.current_fs
        is_complex = np.iscomplexobj(x)

        preset = self._get_preset()

        nperseg = preset["nperseg"]

        if len(x) < nperseg:
            nperseg = len(x)

        noverlap = int(nperseg * preset["overlap"])

        if noverlap >= nperseg:
            noverlap = nperseg - 1

        if is_complex:
            f, t, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd',
                return_onesided=False
            )
            f = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx, axes=0)
        else:
            f, t, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd'
            )

        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        vmin = np.percentile(Sxx_dB, preset["vmin_pct"])
        vmax = np.percentile(Sxx_dB, preset["vmax_pct"])

        im = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud',
                          cmap=self.cmap_combo.currentText(),
                          vmin=vmin, vmax=vmax)

        self.figure.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)')

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if self.current_modulation:
            ax.set_title(f"{self.current_modulation} Spectrogram")
        else:
            ax.set_title("Spectrogram")

        if is_complex:
            ax.set_ylim(-fs/2, fs/2)
        else:
            ax.set_ylim(0, fs/2)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self.figure.tight_layout()
        self.canvas.draw()
