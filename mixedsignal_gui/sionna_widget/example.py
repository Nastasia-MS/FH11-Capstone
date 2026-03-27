#!/usr/bin/env python3
"""Standalone example: embed SionnaWidget in a QMainWindow.

Demonstrates:
- Creating and embedding the widget
- Connecting to paths_computed / augmentation_ready signals
- Running a simple pipeline (channel + AWGN) and visualising stages
"""

import sys
import os

# Allow running as `python example.py` from inside the sionna_widget directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import Slot

from sionna_widget import SionnaWidget, SionnaChannelAugmentation, ChannelParameters


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sionna Widget Demo")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Drop-in widget
        self.sionna = SionnaWidget()
        layout.addWidget(self.sionna)

        # Connect public signals
        self.sionna.paths_computed.connect(self.on_paths_computed)
        self.sionna.augmentation_ready.connect(self.on_augmentation_ready)
        self.sionna.scene_loaded.connect(
            lambda p: self.statusBar().showMessage(f"Scene loaded: {p}")
        )
        self.sionna.error_occurred.connect(
            lambda e: self.statusBar().showMessage(f"Error: {e}")
        )

        self.statusBar().showMessage("Ready - load a scene to begin")

    @Slot(object)
    def on_paths_computed(self, params: ChannelParameters):
        self.statusBar().showMessage(
            f"Paths computed: {params.num_paths} paths"
        )
        print("Channel parameters:", params.to_dict())

    @Slot(object)
    def on_augmentation_ready(self, aug: SionnaChannelAugmentation):
        """Run a demo pipeline and visualise the stages."""
        # Generate a test signal: QPSK at 10 MHz sample rate
        fs = 10e6
        N = 4096
        t = np.arange(N) / fs
        symbols = np.exp(1j * np.pi / 4 * (2 * np.random.randint(0, 4, N) + 1))
        input_signal = symbols

        # Stage 1: input
        stage_names = ["Input"]
        stage_signals = [input_signal.copy()]

        # Stage 2: channel augmentation
        after_channel = aug.apply(input_signal, fs)
        stage_names.append("After Channel")
        stage_signals.append(after_channel)

        # Stage 3: AWGN
        snr_db = 20.0
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(after_channel))
            + 1j * np.random.randn(len(after_channel))
        )
        after_awgn = after_channel + noise
        stage_names.append("After AWGN (20 dB)")
        stage_signals.append(after_awgn)

        # Visualise
        self.sionna.visualize_pipeline(stage_names, stage_signals, fs)
        self.statusBar().showMessage(
            f"Pipeline visualised: {len(stage_names)} stages"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DemoWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
