"""
Comparison visualization widget for clean vs augmented signals.

Uses the same DSP routines as the Waveform Generation tab
(via signal_utils) so that constellations, IQ trajectories,
and spectrograms are visually consistent.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy import signal

from widgets.signal_utils import demodulate_to_symbols, extract_fsk_iq, extract_fhss_iq


# Modulation-aware spectrogram presets
_SPECTROGRAM_PRESETS = {
    "PAM":  {"nperseg": 512,  "overlap": 0.70, "window": "hann",     "vmin_pct": 10, "vmax_pct": 95},
    "QAM":  {"nperseg": 1024, "overlap": 0.75, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 95},
    "PSK":  {"nperseg": 1024, "overlap": 0.75, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 95},
    "ASK":  {"nperseg": 512,  "overlap": 0.70, "window": "hann",     "vmin_pct": 10, "vmax_pct": 95},
    "FSK":  {"nperseg": 2048, "overlap": 0.85, "window": "blackman", "vmin_pct": 3,  "vmax_pct": 97},
    "OFDM": {"nperseg": 2048, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 90},
    "FHSS": {"nperseg": 2048, "overlap": 0.85, "window": "blackman", "vmin_pct": 3,  "vmax_pct": 97},
}
_DEFAULT_PRESET = {"nperseg": 1024, "overlap": 0.75, "window": "hann", "vmin_pct": 5, "vmax_pct": 95}


class ComparisonWidget(QWidget):
    """Widget to display clean vs augmented signal comparison with overlays"""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.clear_plots()

    def clear_plots(self):
        """Clear all subplots"""
        self.figure.clear()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def plot_comparison(self, clean_signal, augmented_signal, fs,
                        fc=None, sps=None, modulation=None,
                        M=None, alpha=0.35, span=8,
                        pulse_shape='rrc', nsymb=None):
        """
        Plot comparison between clean and augmented signals.

        Parameters match the metadata saved by the Waveform Generation tab
        so that the Channel tab can forward them directly.
        """
        self.figure.clear()

        any_complex = np.iscomplexobj(clean_signal) or np.iscomplexobj(augmented_signal)

        # ---- 1. Time Domain Overlay ----
        ax1 = self.figure.add_subplot(3, 2, 1)
        self._plot_time_domain(ax1, clean_signal, augmented_signal, fs)

        # ---- 2. Power Spectrum Overlay (Welch PSD) ----
        ax2 = self.figure.add_subplot(3, 2, 2)
        self._plot_psd(ax2, clean_signal, augmented_signal, fs, any_complex)

        # ---- 3 & 4. Spectrograms ----
        ax3 = self.figure.add_subplot(3, 2, 3)
        ax4 = self.figure.add_subplot(3, 2, 4)
        self._plot_spectrogram(ax3, clean_signal, fs, any_complex, modulation, 'Clean Signal Spectrogram')
        self._plot_spectrogram(ax4, augmented_signal, fs, any_complex, modulation, 'Augmented Signal Spectrogram')

        # ---- 5. Constellation / IQ Trajectory ----
        ax5 = self.figure.add_subplot(3, 2, 5)
        self._plot_constellation(ax5, clean_signal, augmented_signal, fs,
                                 fc=fc, sps=sps, modulation=modulation,
                                 M=M, alpha=alpha, span=span,
                                 pulse_shape=pulse_shape, nsymb=nsymb)

        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Subplot helpers
    # ------------------------------------------------------------------
    def _plot_time_domain(self, ax, clean, augmented, fs):
        n_samples = min(200, len(clean))
        t = np.arange(n_samples) / fs * 1000  # ms

        if np.iscomplexobj(augmented):
            ax.plot(t, np.real(clean[:n_samples]), 'b-', linewidth=1.5, label='Clean I', alpha=0.8)
            ax.plot(t, np.imag(clean[:n_samples]), 'b--', linewidth=1.0, label='Clean Q', alpha=0.6)
            ax.plot(t, np.real(augmented[:n_samples]), 'r-', linewidth=0.8, label='Aug I', alpha=0.6)
            ax.plot(t, np.imag(augmented[:n_samples]), 'r--', linewidth=0.6, label='Aug Q', alpha=0.4)
        else:
            ax.plot(t, clean[:n_samples], 'b-', linewidth=1.5, label='Clean', alpha=0.8)
            ax.plot(t, augmented[:n_samples], 'r-', linewidth=0.8, label='Augmented', alpha=0.6)

        ax.set_title('Time Domain: Clean vs Augmented')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_psd(self, ax, clean, augmented, fs, any_complex):
        if any_complex:
            _, psd_clean = signal.welch(clean, fs, nperseg=1024, return_onesided=False)
            _, psd_aug = signal.welch(augmented, fs, nperseg=1024, return_onesided=False)
            psd_clean = np.fft.fftshift(psd_clean)
            psd_aug = np.fft.fftshift(psd_aug)
            f_psd = np.fft.fftshift(np.fft.fftfreq(len(psd_clean), d=1.0 / fs))
            ax.semilogy(f_psd / 1000, psd_clean, 'b-', linewidth=2, label='Clean')
            ax.semilogy(f_psd / 1000, psd_aug, 'r-', linewidth=1, alpha=0.7, label='Augmented')
        else:
            f_clean, psd_clean = signal.welch(clean, fs, nperseg=1024)
            f_aug, psd_aug = signal.welch(augmented, fs, nperseg=1024)
            ax.semilogy(f_clean / 1000, psd_clean, 'b-', linewidth=2, label='Clean')
            ax.semilogy(f_aug / 1000, psd_aug, 'r-', linewidth=1, alpha=0.7, label='Augmented')

        ax.set_title('Power Spectrum: Clean vs Augmented')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_spectrogram(self, ax, x, fs, is_complex, modulation, title):
        preset = _SPECTROGRAM_PRESETS.get(modulation, _DEFAULT_PRESET)
        nperseg = preset["nperseg"]

        if len(x) < nperseg:
            nperseg = len(x)
        noverlap = int(nperseg * preset["overlap"])
        if noverlap >= nperseg:
            noverlap = nperseg - 1

        if is_complex:
            f, t_s, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd',
                return_onesided=False,
            )
            f = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx, axes=0)
        else:
            f, t_s, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd',
            )

        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        vmin = np.percentile(Sxx_dB, preset["vmin_pct"])
        vmax = np.percentile(Sxx_dB, preset["vmax_pct"])

        im = ax.pcolormesh(
            t_s * 1000, f / 1000, Sxx_dB,
            shading='gouraud', cmap='viridis',
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (kHz)')
        self.figure.colorbar(im, ax=ax, label='dB/Hz')

    def _plot_constellation(self, ax, clean, augmented, fs,
                            fc, sps, modulation, M, alpha, span,
                            pulse_shape, nsymb):
        """Plot constellation or IQ trajectory for both clean and augmented."""
        if sps is None or modulation is None:
            ax.text(0.5, 0.5, 'Constellation not available\n(requires sps and modulation)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        sps = int(sps)

        try:
            if modulation == 'FSK':
                if M is None:
                    M = 4
                I_c, Q_c = extract_fsk_iq(clean, fs, fc, sps, M)
                I_a, Q_a = extract_fsk_iq(augmented, fs, fc, sps, M)
                ax.scatter(I_c, Q_c, c='blue', s=8, alpha=0.4, label='Clean')
                ax.scatter(I_a, Q_a, c='red', s=8, alpha=0.25, label='Augmented')
                ax.set_title(f'{int(M)}-FSK IQ Trajectory: Clean vs Augmented')

            elif modulation == 'FHSS':
                if M is None:
                    M = 4
                I_c, Q_c = extract_fhss_iq(clean, fs, fc, sps, M)
                I_a, Q_a = extract_fhss_iq(augmented, fs, fc, sps, M)
                ax.scatter(I_c, Q_c, c='blue', s=8, alpha=0.4, label='Clean')
                ax.scatter(I_a, Q_a, c='red', s=8, alpha=0.25, label='Augmented')
                ax.set_title(f'FHSS IQ Trajectory: Clean vs Augmented')

            else:
                # PAM / QAM / PSK — matched-filter demodulation
                sym_clean = demodulate_to_symbols(
                    clean, fs, fc, sps, alpha=alpha, span=span,
                    pulse_shape=pulse_shape, nsymb=nsymb,
                )
                sym_aug = demodulate_to_symbols(
                    augmented, fs, fc, sps, alpha=alpha, span=span,
                    pulse_shape=pulse_shape, nsymb=nsymb,
                )
                ax.scatter(np.real(sym_clean), np.imag(sym_clean),
                           c='blue', s=15, alpha=0.5, label='Clean')
                ax.scatter(np.real(sym_aug), np.imag(sym_aug),
                           c='red', s=15, alpha=0.3, label='Augmented')
                if M is not None:
                    ax.set_title(f'{int(M)}-{modulation} Constellation: Clean vs Augmented')
                else:
                    ax.set_title(f'{modulation} Constellation: Clean vs Augmented')

        except Exception as e:
            ax.text(0.5, 0.5, f'Constellation error:\n{str(e)}',
                    ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
