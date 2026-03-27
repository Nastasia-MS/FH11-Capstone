"""Pipeline visualization panel: time-domain, PSD, and IQ constellation per stage."""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MiniCanvas(FigureCanvasQTAgg):
    """Small matplotlib figure for embedding in the pipeline panel."""

    def __init__(self, parent=None, width=2.5, height=1.8, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor("#1e1e1e")
        super().__init__(self.fig)


class StagePlotGroup(QWidget):
    """Three side-by-side plots for one pipeline stage: time, PSD, IQ."""

    def __init__(self, stage_name: str, signal: np.ndarray, fs: float,
                 parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel(f"<b>{stage_name}</b>")
        label.setStyleSheet("color: #cccccc; font-size: 12px;")
        layout.addWidget(label)

        plots_row = QHBoxLayout()

        # -- Time domain ---------------------------------------------------
        self._time_canvas = MiniCanvas()
        self._plot_time(signal, fs)
        plots_row.addWidget(self._time_canvas)

        # -- PSD -----------------------------------------------------------
        self._psd_canvas = MiniCanvas()
        self._plot_psd(signal, fs)
        plots_row.addWidget(self._psd_canvas)

        # -- IQ constellation ---------------------------------------------
        self._iq_canvas = MiniCanvas()
        self._plot_iq(signal)
        plots_row.addWidget(self._iq_canvas)

        layout.addLayout(plots_row)

    # ── Plot helpers ─────────────────────────────────────

    def _plot_time(self, signal: np.ndarray, fs: float):
        ax = self._time_canvas.fig.add_subplot(111)
        ax.set_facecolor("#2a2a2a")

        N = len(signal)
        t_us = np.arange(N) / fs * 1e6  # microseconds

        if np.iscomplexobj(signal):
            ax.plot(t_us, signal.real, color="#4fc3f7", linewidth=0.6,
                    label="I", alpha=0.85)
            ax.plot(t_us, signal.imag, color="#ff8a65", linewidth=0.6,
                    label="Q", alpha=0.85)
            ax.legend(fontsize=6, loc="upper right",
                      facecolor="#2a2a2a", edgecolor="#555",
                      labelcolor="white")
        else:
            ax.plot(t_us, signal.real, color="#4fc3f7", linewidth=0.6)

        ax.set_xlabel("Time (us)", color="white", fontsize=7)
        ax.set_ylabel("Amplitude", color="white", fontsize=7)
        ax.set_title("Time Domain", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#555")
        self._time_canvas.fig.tight_layout()

    def _plot_psd(self, signal: np.ndarray, fs: float):
        ax = self._psd_canvas.fig.add_subplot(111)
        ax.set_facecolor("#2a2a2a")

        N = len(signal)
        if N == 0:
            self._psd_canvas.fig.tight_layout()
            return

        spectrum = np.fft.fftshift(np.fft.fft(signal))
        psd = 10 * np.log10(np.abs(spectrum) ** 2 / N + 1e-30)
        freqs_mhz = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / fs)) / 1e6

        ax.plot(freqs_mhz, psd, color="#81c784", linewidth=0.6)
        ax.set_xlabel("Freq (MHz)", color="white", fontsize=7)
        ax.set_ylabel("PSD (dB)", color="white", fontsize=7)
        ax.set_title("Power Spectral Density", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#555")
        self._psd_canvas.fig.tight_layout()

    def _plot_iq(self, signal: np.ndarray):
        ax = self._iq_canvas.fig.add_subplot(111)
        ax.set_facecolor("#2a2a2a")
        ax.set_aspect("equal", adjustable="datalim")

        if not np.iscomplexobj(signal):
            # Real-only: plot on I axis
            sig = signal.astype(np.complex128)
        else:
            sig = signal

        # Downsample for performance
        MAX_PTS = 2000
        if len(sig) > MAX_PTS:
            indices = np.linspace(0, len(sig) - 1, MAX_PTS, dtype=int)
            sig = sig[indices]

        ax.scatter(sig.real, sig.imag, s=1, color="#ce93d8", alpha=0.6)
        ax.set_xlabel("I", color="white", fontsize=7)
        ax.set_ylabel("Q", color="white", fontsize=7)
        ax.set_title("IQ Constellation", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#555")
        self._iq_canvas.fig.tight_layout()


class PipelineVisualizationPanel(QWidget):
    """Scrollable vertical list of StagePlotGroup widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignTop)

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        self._stage_widgets: list = []

    def display_pipeline_results(self, stage_names: list, stage_signals: list,
                                 fs: float):
        """Clear existing plots and rebuild for each stage.

        Args:
            stage_names: list of str labels for each stage
            stage_signals: list of np.ndarray (the signal after each stage)
            fs: sample rate in Hz
        """
        self.clear()
        for name, sig in zip(stage_names, stage_signals):
            group = StagePlotGroup(name, np.asarray(sig), fs)
            self._container_layout.addWidget(group)
            self._stage_widgets.append(group)

    def clear(self):
        """Remove all stage plot groups."""
        for w in self._stage_widgets:
            self._container_layout.removeWidget(w)
            w.deleteLater()
        self._stage_widgets.clear()
