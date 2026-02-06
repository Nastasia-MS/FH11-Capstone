from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QSpinBox, QFrame, 
                               QSlider, QGridLayout, QTabWidget, QDoubleSpinBox, QLineEdit)
from PySide6.QtCore import Qt

from widgets.constellation import ConstellationWidget
from widgets.power_spectrum import PowerSpectrumWidget
from widgets.waveform_plots import PlottingWidget, FreqDomainPlot, IQDomainPlot, SpectrogramPlot
import numpy as np


class WaveformSelectionTab(QWidget):
    """Waveform configuration and visualization tab"""
    def __init__(self, matlab_engine, parent=None):
        super().__init__(parent)
        self.matlab = matlab_engine
        
        # Core parameters (defaults chosen to be valid: fc < fs/2)
        self.fc = 1e6 # Hz
        self.fs = 8e6
        self.var = 1.0
        self.alpha = 0.35
        self.Tsymb = 1e-6 # seconds
        self.M = 4
        self.Nsymb = 256
        self.span = 10 # symbols
        self.modulation = "PAM"

        self.current_data = None
        self.current_fs = None
        self.current_modulation = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        #layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - RF Signal Configuration
        left_panel = self.create_configuration_panel()
        layout.addWidget(left_panel, 1)
        
        right_panel = self.create_visualizations_panel()
        layout.addWidget(right_panel, 2)
    
    def create_configuration_panel(self):
        """Create the RF signal configuration panel"""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        
        # Title
        #title_layout = QVBoxLayout()
        #title = QLabel("📡 RF Signal Configuration")
        #title.setProperty("class", "section-title")
        #subtitle = QLabel("Configure signal generation parameters")
        #subtitle.setProperty("class", "section-subtitle")
        #title_layout.addWidget(title)
        #title_layout.addWidget(subtitle)
        #title_layout.setSpacing(4)
        #layout.addLayout(title_layout)

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
        self.fs_spin.setValue(self.fs / 1e6)
        self.fs_spin.valueChanged.connect(lambda v: setattr(self, "fs", v * 1e6))
        layout.addWidget(self.fs_spin)

        # fc
        layout.addWidget(QLabel("Carrier Frequency fc (MHz)"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0.1, 200)
        self.fc_spin.setValue(self.fc / 1e6)
        self.fc_spin.valueChanged.connect(lambda v: setattr(self, "fc", v * 1e6))
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
        self.tsymb_spin.setValue(self.Tsymb * 1e6)
        self.tsymb_spin.valueChanged.connect(lambda v: setattr(self, "Tsymb", v * 1e-6))
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

        #layout.addStretch()


        generate_btn = QPushButton("▶ Generate Dataset")
        generate_btn.clicked.connect(self.generate_dataset)
        layout.addWidget(generate_btn)
        
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
        ft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1 / fs) * 1e-6

        sps = int(fs * self.Tsymb)

        self.waveform_plot.plot_data(t, np.real(data))
        self.freq_plot.plot_data(freqs, np.abs(ft))
        self.constellation_plot.plot_data(
        data=data,
        fs=fs,
        fc=self.fc,
        sps=sps,
        M=self.M,
        modulation=self.current_modulation,
        nsymb=self.Nsymb,
        eng=self.matlab.eng,
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


        from backend.waveform_pipeline import WaveformPipeline
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
            pulse_shape=pulse_shape
        )

        self.current_data = result["signal"]
        self.current_fs = fs
        self.current_modulation = modulation

        self.update_waveform_plots()


    def save_configuration(self):
        """Handle save configuration button click"""
        print("Save Configuration clicked")
        # In real implementation: save to JSON
    
    def export_samples(self):
        """Handle export samples button click"""
        print("Export Samples clicked")
        # In real implementation: export data