from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QFrame, QScrollArea,
                               QSlider, QTabWidget, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt

from mixedsignal_gui.widgets.waveform_plots import PlottingWidget, FreqDomainPlot, IQDomainPlot, SpectrogramPlot
import numpy as np
from datetime import datetime


class WaveformSelectionTab(QWidget):
    """Waveform configuration and visualization tab"""
    def __init__(self, matlab_engine, dataset_manager, parent=None):
        super().__init__(parent)
        self.matlab = matlab_engine
        self.dataset_manager = dataset_manager

        # Core parameters
        self.fc = 1e6 # Hz
        self.fs = 8e6
        self.var = 1.0
        self.alpha = 0.35
        self.Tsymb = 1e-6
        self.M = 4
        self.Nsymb = 256
        self.span = 10
        self.modulation = "PAM"
        self.output_type = "passband"

        self.current_data = None
        self.current_fs = None
        self.current_modulation = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        left_panel = self.create_configuration_panel()
        left_scroll = QScrollArea()
        left_scroll.setObjectName("card")
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll, 1)
        
        right_panel = self.create_visualizations_panel()
        layout.addWidget(right_panel, 2)
    
    def create_configuration_panel(self):
        """Create the RF signal configuration panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        # Waveform (Modulation Type)
        layout.addWidget(QLabel("Waveform"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(["PAM", "QAM", "PSK", "FSK", "FHSS"])
        layout.addWidget(self.waveform_combo)

        # fs
        layout.addWidget(QLabel("Sampling Frequency fs (MHz)"))
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.1, 1000)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setValue(self.fs / 1e6)  # Convert to MHz for display
        self.fs_spin.valueChanged.connect(lambda v: setattr(self, "fs", v * 1e6))  # Convert back to Hz
        layout.addWidget(self.fs_spin)

        # fc
        layout.addWidget(QLabel("Carrier Frequency fc (MHz)"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0.1, 200)
        self.fc_spin.setValue(self.fc / 1e6)  # Convert to MHz for display
        self.fc_spin.valueChanged.connect(lambda v: setattr(self, "fc", v * 1e6))  # Convert back to Hz
        layout.addWidget(self.fc_spin)

        # var
        layout.addWidget(QLabel("Noise Variance"))
        self.var_spin = QDoubleSpinBox()
        self.var_spin.setRange(0.0, 10.0)
        self.var_spin.setSingleStep(0.1)
        self.var_spin.setValue(self.var)
        self.var_spin.valueChanged.connect(lambda v: setattr(self, "var", v))
        layout.addWidget(self.var_spin)

        # alpha
        layout.addWidget(QLabel("RRC Roll-off α"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.valueChanged.connect(lambda v: setattr(self, "alpha", v))
        layout.addWidget(self.alpha_spin)


        # Tsymb
        layout.addWidget(QLabel("Symbol Period Tsymb (µs)"))
        self.tsymb_spin = QDoubleSpinBox()
        self.tsymb_spin.setRange(0.01, 100.0)
        self.tsymb_spin.setValue(self.Tsymb * 1e6)  # Convert to microseconds for display
        self.tsymb_spin.valueChanged.connect(lambda v: setattr(self, "Tsymb", v * 1e-6))  # Convert back to seconds
        layout.addWidget(self.tsymb_spin)


        # M
        layout.addWidget(QLabel("Modulation Order M"))
        self.M_spin = QDoubleSpinBox()
        self.M_spin.setRange(2, 256)
        self.M_spin.setValue(self.M)
        self.M_spin.valueChanged.connect(lambda v: setattr(self, "M", v))
        layout.addWidget(self.M_spin)


        # Nsymb
        layout.addWidget(QLabel("Number of Symbols"))
        self.nsymb_spin = QDoubleSpinBox()
        self.nsymb_spin.setRange(16, 10000)
        self.nsymb_spin.setValue(self.Nsymb)
        self.nsymb_spin.valueChanged.connect(lambda v: setattr(self, "Nsymb", v))
        layout.addWidget(self.nsymb_spin)


        # span
        layout.addWidget(QLabel("Pulse Span (symbols)"))
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setRange(2, 50)
        self.span_spin.setValue(self.span)
        self.span_spin.valueChanged.connect(lambda v: setattr(self, "span", v))
        layout.addWidget(self.span_spin)

        # Pulse Shape
        pulse_label = QLabel("Pulse Shape")
        layout.addWidget(pulse_label)

        self.pulse_shape_combo = QComboBox()
        self.pulse_shape_combo.addItems(["rrc", "rect"])
        self.pulse_shape_combo.setCurrentText("rrc")
        layout.addWidget(self.pulse_shape_combo)

        # Output Type
        output_type_label = QLabel("Output Type")
        layout.addWidget(output_type_label)

        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(["Passband (Real)", "Baseband (Complex IQ)"])
        self.output_type_combo.currentIndexChanged.connect(
            lambda idx: setattr(self, "output_type", "passband" if idx == 0 else "baseband")
        )
        layout.addWidget(self.output_type_combo)

        generate_btn = QPushButton("▶ Generate Dataset")
        generate_btn.clicked.connect(self.generate_dataset)
        layout.addWidget(generate_btn)

        save_btn = QPushButton("💾 Save to Dataset Manager")
        save_btn.clicked.connect(self.save_to_dataset_manager)
        layout.addWidget(save_btn)
        
        # Batch generation section
        batch_label = QLabel("Batch Generation")
        batch_label.setProperty("class", "section-title")
        layout.addWidget(batch_label)
        
        # Number of samples input
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Samples per modulation:"))
        self.batch_samples_spin = QSpinBox()
        self.batch_samples_spin.setRange(1, 1000)
        self.batch_samples_spin.setValue(10)
        batch_layout.addWidget(self.batch_samples_spin)
        layout.addLayout(batch_layout)
        
        # Batch generate button
        batch_btn = QPushButton("📦 Batch Generate")
        batch_btn.clicked.connect(self.batch_generate)
        layout.addWidget(batch_btn)
        
        return panel
    
    def create_slider_control(self, label, value, unit, min_val, max_val, attr_name):
        """Create a slider control with label and value display"""
        container = QVBoxLayout()
        container.setSpacing(8)
        
        # Label and value
        header = QHBoxLayout()
        label_widget = QLabel(label)
        value_label = QLabel(f"{value} {unit}")
        value_label.setProperty("class", "stat-value")
        value_label.setMinimumHeight(24)
        header.addWidget(label_widget)
        header.addStretch()
        header.addWidget(value_label)
        container.addLayout(header)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        
        def update_value(v):
            value_label.setText(f"{v} {unit}")
            setattr(self, attr_name, v)
            self.update_waveform_plots()
        
        slider.valueChanged.connect(update_value)
        container.addWidget(slider)
        
        return container
    
    def create_visualizations_panel(self):
        """Create the visualizations panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        
        self.plot_tabs = QTabWidget()
        self.waveform_plot = PlottingWidget()
        self.freq_plot = FreqDomainPlot()
        self.constellation_plot = IQDomainPlot()
        self.spectrogram_plot = SpectrogramPlot()

        self.plot_tabs.addTab(self.waveform_plot, "Waveform")
        self.plot_tabs.addTab(self.freq_plot, "Frequency")
        self.plot_tabs.addTab(self.constellation_plot, "Constellation")
        self.plot_tabs.addTab(self.spectrogram_plot, "Spectrogram")

        layout.addWidget(self.plot_tabs)
        
        return panel

    def update_waveform_plots(self):
        if self.current_data is None:
            return

        data = self.current_data
        fs = self.current_fs

        t = np.arange(len(data)) / fs * 1e6
        sps = int(fs * self.Tsymb)

        # Time domain: for complex data plot I and Q; for real plot as-is
        if np.iscomplexobj(data):
            self.waveform_plot.plot_data(t, np.real(data), np.imag(data))
        else:
            self.waveform_plot.plot_data(t, np.real(data))

        # Frequency domain: for complex use fftshift for centered spectrum
        if np.iscomplexobj(data):
            ft = np.fft.fftshift(np.fft.fft(data))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1 / fs)) * 1e-6
        else:
            ft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1 / fs) * 1e-6

        self.freq_plot.plot_data(freqs, np.abs(ft))

        self.constellation_plot.plot_data(
            data=data,
            fs=fs,
            fc=self.fc,
            sps=sps,
            M=self.M,
            modulation=self.current_modulation,
            nsymb=self.Nsymb,
        )
        self.spectrogram_plot.plot_data(x=data, fs=fs, modulation=self.current_modulation)

    
    def generate_dataset(self):
        modulation = self.waveform_combo.currentText()
        M = self.M
        fs = self.fs            # Hz
        Tsymb = self.Tsymb      # seconds
        Nsymb = self.Nsymb
        fc = self.fc
        alpha = self.alpha
        span = self.span
        var = self.var
        pulse_shape = self.pulse_shape_combo.currentText()

        # Enforce Nyquist
        if fc >= fs / 2:
            raise ValueError(f"Invalid parameters: fc={fc:.2e} Hz must be < fs/2={fs/2:.2e} Hz")

        # Validate: fs * Tsymb must be an integer (samples per symbol)
        sps = fs * Tsymb
        if abs(sps - round(sps)) > 1e-9:
            raise ValueError(f"Invalid parameters: fs * Tsymb = {sps:.6f} must be an integer (samples per symbol)")

        from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
        pipeline = WaveformPipeline(self.matlab)

        result = pipeline.generate(
            fs=fs,
            Tsymb=Tsymb,
            Nsymb=Nsymb,
            fc=fc,
            M=M,
            modulation=modulation,
            var=var,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            output_type=self.output_type
        )

        self.current_data = result["signal"]
        self.current_fs = fs
        self.current_modulation = modulation

        self.update_waveform_plots()


    def save_to_dataset_manager(self):
        """Save the currently generated waveform to the datasets folder."""
        if self.current_data is None:
            print("✗ No waveform generated yet. Generate a dataset first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.current_modulation}_{int(self.M)}_{timestamp}"

        metadata = {
            'source':      'generated',
            'modulation':  self.current_modulation,
            'M':           int(self.M),
            'fc':          self.fc,
            'fs':          self.current_fs,
            'Tsymb':       self.Tsymb,
            'Nsymb':       self.Nsymb,
            'alpha':       self.alpha,
            'span':        self.span,
            'pulse_shape': self.pulse_shape_combo.currentText(),
            'output_type': self.output_type,
            'timestamp':   timestamp,
        }

        self.dataset_manager.save(name, self.current_data, metadata)
        print(f"✓ Saved dataset: {name}")

    
    def batch_generate(self):
        """Generate multiple datasets with random parameters for each modulation."""
        import random

        num_samples = self.batch_samples_spin.value()
        modulations = ["PAM", "QAM", "PSK", "FSK", "FHSS"]

        m_values = {
            "PAM":  [2, 4, 8, 16],
            "QAM":  [4, 16, 64],
            "PSK":  [2, 4, 8],
            "FSK":  [2, 4, 8],
            "FHSS": [4, 8, 16],
        }

        from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
        pipeline = WaveformPipeline(self.matlab)

        total = num_samples * len(modulations)
        count = 0
        print(f"Starting batch generation: {num_samples} x {len(modulations)} = {total} total")

        for modulation in modulations:
            for _ in range(num_samples):
                try:
                    M = random.choice(m_values[modulation])
                    result = pipeline.generate(
                        fs=self.fs,
                        Tsymb=self.Tsymb,
                        Nsymb=self.Nsymb,
                        fc=self.fc,
                        M=M,
                        modulation=modulation,
                        var=self.var,
                        alpha=self.alpha,
                        span=self.span,
                        pulse_shape=self.pulse_shape_combo.currentText(),
                    )
                    data = result["signal"]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    name = f"{modulation}_{M}_{timestamp}"

                    metadata = {
                        "source":      "batch_generated",
                        "modulation":  modulation,
                        "M":           int(M),
                        "fc":          self.fc,
                        "fs":          self.fs,
                        "Tsymb":       self.Tsymb,
                        "Nsymb":       self.Nsymb,
                        "alpha":       self.alpha,
                        "span":        self.span,
                        "pulse_shape": self.pulse_shape_combo.currentText(),
                        "timestamp":   timestamp,
                    }

                    self.dataset_manager.save(name, data, metadata)
                    count += 1
                    if count % 10 == 0:
                        print(f"  Progress: {count}/{total}")

                except Exception as e:
                    print(f"X Error generating {modulation} M={M}: {e}")

        print(f"Batch complete: {count}/{total} datasets saved")