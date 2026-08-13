from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QFrame, QScrollArea,
                               QSlider, QTabWidget, QDoubleSpinBox, QSpinBox,
                               QDialog, QCheckBox, QGridLayout, QLineEdit,
                               QMessageBox)
from PySide6.QtCore import Qt

from mixedsignal_gui.widgets.waveform_plots import PlottingWidget, FreqDomainPlot, IQDomainPlot, SpectrogramPlot
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker
from mixedsignal_gui.widgets.modulation_utils import (mark_unavailable_modulations,
                                                     selected_modulation)
import numpy as np
from datetime import datetime


class QuickTestDataDialog(QDialog):
    """Quick dialog to generate small test datasets for ML validation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Quick Test Data")
        self.setMinimumWidth(400)
        self.setMinimumHeight(250)
        
        self.modulations_available = ["PAM", "QAM", "PSK", "FSK", "FHSS",
                                      "LFM", "Barker", "FMCW",
                                      "WiFi", "LTE", "5G_NR", "Zigbee"]
        self.selected_modulations = set(self.modulations_available)  # All selected by default

        # M values for minimal test: just 1-2 M values per modulation
        self.test_m_values = {
            "PAM":    [4],
            "QAM":    [16],
            "PSK":    [4],
            "FSK":    [2],
            "FHSS":   [8],
            "LFM":    [4],
            "Barker": [7],
            "FMCW":   [4],
            "WiFi":   [4],
            "LTE":    [4],
            "5G_NR":  [4],
            "Zigbee": [4],
        }

        self.setup_ui()
        install_wheel_blocker(self)
    
    def setup_ui(self):
        """Create minimal configuration UI."""
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Select modulations for test data:"))
        
        # Checkboxes for each modulation
        self.mod_checkboxes = {}
        for mod in self.modulations_available:
            chk = QCheckBox(mod)
            chk.setChecked(True)
            chk.stateChanged.connect(lambda state, m=mod: self._update_selection(m, state))
            self.mod_checkboxes[mod] = chk
            layout.addWidget(chk)
        
        layout.addSpacing(10)
        layout.addWidget(QLabel("Samples per modulation:"))
        
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 50)
        self.samples_spin.setValue(3)  # Small default for quick generation
        layout.addWidget(self.samples_spin)
        
        layout.addSpacing(10)
        layout.addWidget(QLabel("Note: Uses minimal M values per modulation for quick testing."))
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("Generate Test Data")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
    
    def _update_selection(self, modulation, state):
        """Update selected modulations."""
        if state == 2:  # Checked (Qt.CheckState.Checked)
            self.selected_modulations.add(modulation)
        else:
            self.selected_modulations.discard(modulation)
    
    def get_config(self):
        """Return test data configuration."""
        return {
            'modulations': list(self.selected_modulations),
            'samples_per_mod': self.samples_spin.value(),
            'm_values': self.test_m_values,
        }


class BatchGenerationConfigDialog(QDialog):
    """Dialog to configure per-modulation parameters for batch generation."""
    
    def __init__(self, global_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Generation Configuration")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self.global_params = global_params
        self.modulations = ["PAM", "QAM", "PSK", "FSK", "FHSS",
                            "LFM", "Barker", "FMCW", "WiFi", "LTE", "5G_NR",
                            "Zigbee"]

        # Default M values for each modulation
        self.default_m_values = {
            "PAM":    [2, 4, 8, 16],
            "QAM":    [4, 16, 64],
            "PSK":    [2, 4, 8],
            "FSK":    [2, 4, 8],
            "FHSS":   [4, 8, 16],
            "LFM":    [2, 4, 8],
            "Barker": [5, 7, 11, 13],
            "FMCW":   [2, 4, 8],
            "WiFi":   [4],
            "LTE":    [4],
            "5G_NR":  [4],
            "Zigbee": [4],
        }

        # Sensible defaults for waveforms that need higher fs / different fc.
        # The standards rows default to 30.72 Msps: LTE R.9 and the 20 MHz NR
        # carrier are natively 30.72 Msps, so no resampling happens there, and
        # keeping every standards class on one rate makes their occupied
        # bandwidths directly comparable.
        self._waveform_defaults = {
            'LFM':    {'fs_override': 1e6, 'fc_override': 200e3, 'Tsymb_override': 1e-4},
            'Barker': {'fs_override': 1e6, 'fc_override': 200e3, 'Tsymb_override': 5e-5},
            'FMCW':   {'fs_override': 1e6, 'fc_override': 200e3, 'Tsymb_override': 1e-4},
            'WiFi':   {'fs_override': 30.72e6, 'fc_override': 10e6},
            'LTE':    {'fs_override': 30.72e6, 'fc_override': 10e6},
            '5G_NR':  {'fs_override': 30.72e6, 'fc_override': 10e6},
            'Zigbee': {'fs_override': 30.72e6, 'fc_override': 10e6},
        }

        # Per-modulation configurations: {mod: {'enabled': bool, 'M': [M_values], 'fs': val, 'fc': val, ...}}
        self.config = {}
        for mod in self.modulations:
            wf_defaults = self._waveform_defaults.get(mod, {})
            self.config[mod] = {
                'enabled': True,
                'M_values': self.default_m_values[mod].copy(),
                'fs_override': wf_defaults.get('fs_override', None),
                'fc_override': wf_defaults.get('fc_override', None),
                'Tsymb_override': wf_defaults.get('Tsymb_override', None),
                'num_samples': 10,
            }
        
        self.setup_ui()
        install_wheel_blocker(self)
    
    def setup_ui(self):
        """Create the configuration UI."""
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Configure per-modulation parameters for batch generation"))
        
        # Scroll area for modulation configs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Create config for each modulation
        for mod in self.modulations:
            mod_frame = self.create_modulation_frame(mod)
            scroll_layout.addWidget(mod_frame)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("Start Batch Generation")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
    
    def create_modulation_frame(self, modulation):
        """Create configuration frame for one modulation type."""
        frame = QFrame()
        frame.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 4px; } QFrame > * { margin: 4px; }")
        layout = QGridLayout(frame)
        
        row = 0
        
        # Enable checkbox
        enable_chk = QCheckBox(modulation)
        enable_chk.setChecked(True)
        enable_chk.stateChanged.connect(lambda: self._update_config(modulation, 'enabled', enable_chk.isChecked()))
        layout.addWidget(enable_chk, row, 0, 1, 2)
        
        row += 1
        
        # M values (comma-separated)
        layout.addWidget(QLabel("M values (comma-separated):"), row, 0)
        m_edit = QLineEdit()
        m_edit.setText(", ".join(map(str, self.default_m_values[modulation])))
        m_edit.editingFinished.connect(lambda: self._parse_m_values(modulation, m_edit.text()))
        layout.addWidget(m_edit, row, 1)
        
        row += 1
        
        # Num samples
        layout.addWidget(QLabel("Samples per M value:"), row, 0)
        samples_spin = QSpinBox()
        samples_spin.setRange(1, 100)
        samples_spin.setValue(10)
        samples_spin.valueChanged.connect(lambda v: self._update_config(modulation, 'num_samples', v))
        layout.addWidget(samples_spin, row, 1)
        
        row += 1
        
        # Override fs (optional)
        layout.addWidget(QLabel("fs override (Hz, leave blank for global):"), row, 0)
        fs_edit = QLineEdit()
        fs_edit.setPlaceholderText("e.g., 48000")
        if self.config[modulation]['fs_override'] is not None:
            fs_edit.setText(str(self.config[modulation]['fs_override']))
        fs_edit.editingFinished.connect(lambda: self._parse_float(modulation, 'fs_override', fs_edit.text()))
        layout.addWidget(fs_edit, row, 1)

        row += 1

        # Override fc (optional)
        layout.addWidget(QLabel("fc override (Hz, leave blank for global):"), row, 0)
        fc_edit = QLineEdit()
        fc_edit.setPlaceholderText("e.g., 6000")
        if self.config[modulation]['fc_override'] is not None:
            fc_edit.setText(str(self.config[modulation]['fc_override']))
        fc_edit.editingFinished.connect(lambda: self._parse_float(modulation, 'fc_override', fc_edit.text()))
        layout.addWidget(fc_edit, row, 1)

        row += 1

        # Override Tsymb (optional)
        layout.addWidget(QLabel("Tsymb override (seconds, leave blank for global):"), row, 0)
        tsymb_edit = QLineEdit()
        tsymb_edit.setPlaceholderText("e.g., 0.001")
        if self.config[modulation]['Tsymb_override'] is not None:
            tsymb_edit.setText(str(self.config[modulation]['Tsymb_override']))
        tsymb_edit.editingFinished.connect(lambda: self._parse_float(modulation, 'Tsymb_override', tsymb_edit.text()))
        layout.addWidget(tsymb_edit, row, 1)
        
        return frame
    
    def _update_config(self, modulation, key, value):
        """Update config for a modulation."""
        self.config[modulation][key] = value
    
    def _parse_m_values(self, modulation, text):
        """Parse comma-separated M values."""
        try:
            values = [int(v.strip()) for v in text.split(',') if v.strip()]
            if values:
                self.config[modulation]['M_values'] = values
        except ValueError:
            pass  # Ignore invalid input
    
    def _parse_float(self, modulation, key, text):
        """Parse optional float override."""
        try:
            if text.strip():
                self.config[modulation][key] = float(text.strip())
            else:
                self.config[modulation][key] = None
        except ValueError:
            pass  # Ignore invalid input
    
    def get_config(self):
        """Return the configured parameters."""
        return self.config


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
        self.current_baseband_symbols = None
        
        self.setup_ui()
        install_wheel_blocker(self)
    
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
        self.waveform_combo.addItems(["PAM", "QAM", "PSK", "FSK", "FHSS",
                                          "LFM", "Barker", "FMCW",
                                          "WiFi", "LTE", "5G_NR", "Zigbee"])
        mark_unavailable_modulations(self.waveform_combo, self.matlab)
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
        
        # Quick test data button
        test_btn = QPushButton("⚡ Quick Test Data")
        test_btn.clicked.connect(self.generate_quick_test_data)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(batch_btn)
        btn_layout.addWidget(test_btn)
        layout.addLayout(btn_layout)
        
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
        sps = int(fs * self.Tsymb)

        t = np.arange(len(data)) / fs * 1e6

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
            baseband_symbols=self.current_baseband_symbols,
        )
        self.spectrogram_plot.plot_data(x=data, fs=fs, modulation=self.current_modulation)

    
    def generate_dataset(self):
        """Generate one waveform and refresh the plots.

        Everything is wrapped so a failure reaches the user as a dialog.  Qt
        swallows exceptions raised inside a clicked-slot — it prints a
        traceback to the terminal and carries on — so without this an
        unsupported waveform (WiFi without MATLAB, say) looked like the button
        simply did nothing.
        """
        modulation = selected_modulation(self.waveform_combo)
        M = self.M
        fs = self.fs            # Hz
        Tsymb = self.Tsymb      # seconds
        Nsymb = self.Nsymb
        fc = self.fc
        alpha = self.alpha
        span = self.span
        var = self.var
        pulse_shape = self.pulse_shape_combo.currentText()

        try:
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
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Parameters", str(e))
            return
        except Exception as e:
            QMessageBox.critical(
                self, f"Cannot generate {modulation}",
                f"Waveform generation failed:\n\n{e}")
            return

        self.current_data = result["signal"]
        self.current_fs = fs
        self.current_modulation = modulation
        self.current_baseband_symbols = result.get("baseband_symbols")
        # Carried to save_to_dataset_manager(), which runs in a separate method
        self.current_generator = result.get("generator")

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
            'generator':   getattr(self, 'current_generator', None),
        }

        self.dataset_manager.save(name, self.current_data, metadata)
        print(f"✓ Saved dataset: {name}")

    
    def batch_generate(self):
        """Generate datasets with automatic train/test split (75%/25% ratio)."""
        import random

        # Show configuration dialog
        global_params = {
            'fs': self.fs,
            'fc': self.fc,
            'Tsymb': self.Tsymb,
            'var': self.var,
            'alpha': self.alpha,
            'span': self.span,
            'pulse_shape': self.pulse_shape_combo.currentText(),
        }
        
        dialog = BatchGenerationConfigDialog(global_params, self)
        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled
        
        batch_config = dialog.get_config()

        modulations = list(batch_config.keys())
        from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
        pipeline = WaveformPipeline(self.matlab)

        # Calculate total samples for progress tracking
        total = 0
        for modulation in modulations:
            if batch_config[modulation]['enabled']:
                num_samples = batch_config[modulation]['num_samples']
                total += num_samples * len(batch_config[modulation]['M_values'])

        # Train/test split ratio: 0.25 means 75% train, 25% test
        test_ratio = 0.25
        
        count_train = 0
        count_test = 0
        failures = {}          # modulation -> first error seen
        print(f"Starting batch generation with train/test split (75%/25%): {total} total samples")

        for modulation in modulations:
            if not batch_config[modulation]['enabled']:
                print(f"Skipping {modulation} (disabled)")
                continue
            
            mod_config = batch_config[modulation]
            
            # Use modulation-specific overrides or fall back to global values
            fs = mod_config['fs_override'] if mod_config['fs_override'] is not None else self.fs
            fc = mod_config['fc_override'] if mod_config['fc_override'] is not None else self.fc
            Tsymb = mod_config['Tsymb_override'] if mod_config['Tsymb_override'] is not None else self.Tsymb
            
            for M in mod_config['M_values']:
                for sample_idx in range(mod_config['num_samples']):
                    try:
                        result = pipeline.generate(
                            fs=fs,
                            Tsymb=Tsymb,
                            Nsymb=self.Nsymb,
                            fc=fc,
                            M=M,
                            modulation=modulation,
                            var=self.var,
                            alpha=self.alpha,
                            span=self.span,
                            pulse_shape=self.pulse_shape_combo.currentText(),
                            output_type=self.output_type,
                        )
                        data = result["signal"]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        
                        # Determine if this sample goes to train or test set
                        is_test = random.random() < test_ratio
                        split_prefix = "test" if is_test else "train"
                        
                        # Create dataset name with split prefix
                        name = f"{split_prefix}_{modulation}_{M}_{timestamp}"

                        metadata = {
                            "source":      "batch_generated",
                            "data_split":  "test" if is_test else "train",
                            "modulation":  modulation,
                            "M":           int(M),
                            "fc":          fc,
                            "fs":          fs,
                            "Tsymb":       Tsymb,
                            "Nsymb":       self.Nsymb,
                            "alpha":       self.alpha,
                            "span":        self.span,
                            "pulse_shape": self.pulse_shape_combo.currentText(),
                            "output_type": self.output_type,
                            "timestamp":   timestamp,
                            "generator":   result.get("generator"),
                        }

                        self.dataset_manager.save(name, data, metadata)
                        
                        if is_test:
                            count_test += 1
                        else:
                            count_train += 1
                        
                        total_saved = count_train + count_test
                        if total_saved % 10 == 0:
                            print(f"  Progress: {total_saved}/{total} (train: {count_train}, test: {count_test})")

                    except Exception as e:
                        print(f"X Error generating {modulation} M={M}: {e}")
                        # Keep one reason per modulation for the summary dialog;
                        # a failing class usually fails identically every sample.
                        failures.setdefault(modulation, str(e))

        total_saved = count_train + count_test
        print(f"Batch complete: {total_saved}/{total} datasets saved")
        print(f"  Training set: {count_train} samples")
        print(f"  Test set: {count_test} samples")

        # Report in the GUI too.  Previously the only sign that a class had
        # failed was a line on stdout, so a run that quietly dropped WiFi/LTE/
        # 5G_NR looked like a complete success.
        summary = (f"Saved {total_saved} of {total} samples.\n"
                   f"Training set: {count_train}    Test set: {count_test}")
        if failures:
            detail = "\n".join(f"\n• {mod}\n    {reason}" for mod, reason in failures.items())
            QMessageBox.warning(
                self, "Batch generation finished with skipped classes",
                f"{summary}\n\n{len(failures)} modulation(s) produced no data:\n{detail}")
        else:
            QMessageBox.information(self, "Batch generation complete", summary)

    def generate_quick_test_data(self):
        """Generate small test datasets quickly for ML model validation."""
        # Show test data configuration dialog
        dialog = QuickTestDataDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        test_config = dialog.get_config()
        modulations = test_config['modulations']
        samples_per_mod = test_config['samples_per_mod']
        m_values_dict = test_config['m_values']
        
        from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
        pipeline = WaveformPipeline(self.matlab)
        
        total = len(modulations) * samples_per_mod
        count = 0
        
        print(f"Starting quick test data generation: {total} samples ({len(modulations)} modulations)")
        
        for modulation in modulations:
            M = m_values_dict[modulation][0]  # Use first (minimal) M value
            
            for sample_idx in range(samples_per_mod):
                try:
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
                        output_type=self.output_type,
                    )
                    data = result["signal"]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # Use "test_" prefix for easy identification
                    name = f"test_{modulation}_{M}_{timestamp}"
                    
                    metadata = {
                        "source":      "quick_test_generated",
                        "modulation":  modulation,
                        "M":           int(M),
                        "fc":          self.fc,
                        "fs":          self.fs,
                        "Tsymb":       self.Tsymb,
                        "Nsymb":       self.Nsymb,
                        "alpha":       self.alpha,
                        "span":        self.span,
                        "pulse_shape": self.pulse_shape_combo.currentText(),
                        "output_type": self.output_type,
                        "timestamp":   timestamp,
                        "generator":   result.get("generator"),
                    }
                    
                    self.dataset_manager.save(name, data, metadata)
                    count += 1
                    
                except Exception as e:
                    print(f"X Error generating test {modulation} M={M}: {e}")
        
        print(f"Test data generation complete: {count}/{total} samples saved (prefix: 'test_')")