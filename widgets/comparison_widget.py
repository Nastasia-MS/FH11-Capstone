"""
Comparison visualization widget for clean vs augmented signals
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy import signal


class ComparisonWidget(QWidget):
    """Widget to display clean vs augmented signal comparison with overlays"""
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        
        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Clear initial plot
        self.clear_plots()
    
    def clear_plots(self):
        """Clear all subplots"""
        self.figure.clear()
        self.canvas.draw()
    
    def plot_comparison(self, clean_signal, augmented_signal, fs, fc=None, sps=None, modulation=None):
        """
        Plot comparison between clean and augmented signals
        
        Args:
            clean_signal: Original signal
            augmented_signal: Augmented signal
            fs: Sampling frequency
            fc: Carrier frequency (for constellation)
            sps: Samples per symbol (for constellation)
            modulation: Modulation type
        """
        self.figure.clear()
        
        # Create subplot grid: 3 rows x 2 columns
        # Row 1: Time domain (overlay), Power spectrum (overlay)
        # Row 2: Clean spectrogram, Noisy spectrogram
        # Row 3: Constellation (overlay), empty
        
        # 1. Time Domain Overlay
        ax1 = self.figure.add_subplot(3, 2, 1)
        n_samples = min(200, len(clean_signal))  # Zoom in to see detail
        t = np.arange(n_samples) / fs * 1000  # Convert to ms
        ax1.plot(t, clean_signal[:n_samples], 'b-', linewidth=1.5, label='Clean', alpha=0.8)
        ax1.plot(t, augmented_signal[:n_samples], 'r-', linewidth=0.8, label='Augmented', alpha=0.6)
        ax1.set_title('Time Domain: Clean vs Augmented')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Power Spectrum Overlay
        ax2 = self.figure.add_subplot(3, 2, 2)
        f_clean, psd_clean = signal.welch(clean_signal, fs, nperseg=1024)
        f_aug, psd_aug = signal.welch(augmented_signal, fs, nperseg=1024)
        ax2.semilogy(f_clean / 1000, psd_clean, 'b-', linewidth=2, label='Clean')
        ax2.semilogy(f_aug / 1000, psd_aug, 'r-', linewidth=1, alpha=0.7, label='Augmented')
        ax2.set_title('Power Spectrum: Clean vs Augmented')
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Clean Signal Spectrogram
        ax3 = self.figure.add_subplot(3, 2, 3)
        f_spec, t_spec, Sxx_clean = signal.spectrogram(
            clean_signal, fs, nperseg=256, noverlap=200
        )
        im3 = ax3.pcolormesh(
            t_spec * 1000, f_spec / 1000, 
            10 * np.log10(Sxx_clean + 1e-10),
            shading='gouraud', cmap='viridis'
        )
        ax3.set_title('Clean Signal Spectrogram')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Frequency (kHz)')
        self.figure.colorbar(im3, ax=ax3, label='dB')
        
        # 4. Augmented Signal Spectrogram
        ax4 = self.figure.add_subplot(3, 2, 4)
        f_spec, t_spec, Sxx_aug = signal.spectrogram(
            augmented_signal, fs, nperseg=256, noverlap=200
        )
        im4 = ax4.pcolormesh(
            t_spec * 1000, f_spec / 1000,
            10 * np.log10(Sxx_aug + 1e-10),
            shading='gouraud', cmap='viridis'
        )
        ax4.set_title('Augmented Signal Spectrogram')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Frequency (kHz)')
        self.figure.colorbar(im4, ax=ax4, label='dB')
        
        # 5. Constellation Diagram Overlay (if parameters available)
        ax5 = self.figure.add_subplot(3, 2, 5)
        if fc is not None and sps is not None and modulation not in ['FSK', 'FHSS']:
            try:
                I_clean, Q_clean = self._extract_iq(clean_signal, fs, fc, sps)
                I_aug, Q_aug = self._extract_iq(augmented_signal, fs, fc, sps)
                
                # Normalize
                scale = np.max(np.abs(I_clean + 1j * Q_clean))
                if scale > 0:
                    I_clean, Q_clean = I_clean / scale, Q_clean / scale
                    I_aug, Q_aug = I_aug / scale, Q_aug / scale
                
                # Plot
                ax5.scatter(I_clean, Q_clean, c='blue', s=10, alpha=0.5, label='Clean')
                ax5.scatter(I_aug, Q_aug, c='red', s=10, alpha=0.3, label='Augmented')
                ax5.set_title('Constellation: Clean vs Augmented')
                ax5.set_xlabel('In-Phase (I)')
                ax5.set_ylabel('Quadrature (Q)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                ax5.set_aspect('equal')
                ax5.set_xlim(-1.5, 1.5)
                ax5.set_ylim(-1.5, 1.5)
            except Exception as e:
                ax5.text(0.5, 0.5, f'Constellation error:\n{str(e)}', 
                        ha='center', va='center', fontsize=10)
                ax5.set_xlim(0, 1)
                ax5.set_ylim(0, 1)
        else:
            ax5.text(0.5, 0.5, 'Constellation not available\n(requires fc, sps, and non-FSK modulation)', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _extract_iq(self, signal_data, fs, fc, sps):
        """Extract I/Q components from real passband signal"""
        t = np.arange(len(signal_data)) / fs
        
        # Downconvert
        I = signal_data * 2 * np.cos(2 * np.pi * fc * t)
        Q = signal_data * -2 * np.sin(2 * np.pi * fc * t)
        
        # Low-pass filter
        from scipy.signal import butter, filtfilt
        b, a = butter(5, fc / (fs / 2), btype='low')
        I = filtfilt(b, a, I)
        Q = filtfilt(b, a, Q)
        
        # Decimate to symbol rate
        I_symb = I[sps//2::sps]
        Q_symb = Q[sps//2::sps]
        
        return I_symb, Q_symb