from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSlider, QGridLayout, QComboBox,
                               QDoubleSpinBox, QSpinBox, QStackedWidget, QMessageBox,
                               QButtonGroup)
from PySide6.QtCore import Qt

from widgets.toggle_switch import ToggleSwitch
from widgets.noise_spectrum import NoiseSpectrumWidget
from widgets.waveform_plots import PlottingWidget
from backend.augmentation import (AugmentationPipeline, AWGNAugmentation,
                                  ScalarAmplitudeAndPhaseShift, StochasticTDLAugmentation)
import numpy as np
import json
import os


class ChannelNoiseTab(QWidget):
    """Channel configuration and noise settings tab"""

    def __init__(self, dataset_manager):
        super().__init__()
        self.dataset_manager = dataset_manager

        # AWGN params
        self.snr_db = 20.0
        self.amplitude = 1.0
        self.phase_deg = 0.0
        self.awgn_enabled = True
        self.amp_phase_enabled = False

        # Stochastic TDL params
        self.active_subtab = 0  # 0=AWGN, 1=Stochastic
        self.tdl_profile = "A"
        self.delay_spread_ns = 100.0
        self.max_doppler_speed = 3.0
        self.stoch_snr_db = 20.0
        self.stoch_seed = 42
        self.output_num_samples = 1024

        self.setup_ui()
        self.refresh_dataset_list()

        if self.dataset_manager.get_active():
            self.display_original_signal()

    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left panel - Channel Configuration
        left_panel = self.create_controls_panel()
        layout.addWidget(left_panel, 1)

        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        layout.addWidget(right_panel, 2)

    def create_controls_panel(self):
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title
        title = QLabel("Channel Augmentation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Apply channel effects and noise")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Dataset Selection
        dataset_selection = QVBoxLayout()
        dataset_selection.setSpacing(4)
        dataset_label = QLabel("Select Dataset")
        dataset_label.setProperty("class", "section-title")
        dataset_selection.addWidget(dataset_label)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_selection.addWidget(self.dataset_combo)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.setSpacing(6)
        upload_btn = QPushButton("Upload Dataset")
        upload_btn.clicked.connect(self.upload_dataset)
        dataset_buttons.addWidget(upload_btn)

        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.clicked.connect(self.refresh_dataset_list)
        dataset_buttons.addWidget(refresh_btn)
        dataset_selection.addLayout(dataset_buttons)

        layout.addLayout(dataset_selection)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # --- Subtab pills ---
        pill_layout = QHBoxLayout()
        pill_layout.setSpacing(8)

        self.subtab_group = QButtonGroup(self)
        self.subtab_group.setExclusive(True)

        self.awgn_pill = QPushButton("AWGN")
        self.awgn_pill.setObjectName("subtabPill")
        self.awgn_pill.setCheckable(True)
        self.awgn_pill.setChecked(True)

        self.stoch_pill = QPushButton("Stochastic Channel")
        self.stoch_pill.setObjectName("subtabPill")
        self.stoch_pill.setCheckable(True)

        self.subtab_group.addButton(self.awgn_pill, 0)
        self.subtab_group.addButton(self.stoch_pill, 1)

        pill_layout.addWidget(self.awgn_pill)
        pill_layout.addWidget(self.stoch_pill)
        pill_layout.addStretch()
        layout.addLayout(pill_layout)

        # --- Stacked widget ---
        self.subtab_stack = QStackedWidget()
        self.subtab_stack.addWidget(self._create_awgn_page())
        self.subtab_stack.addWidget(self._create_stochastic_page())
        layout.addWidget(self.subtab_stack)

        self.subtab_group.idClicked.connect(self._on_subtab_changed)

        layout.addStretch()

        # Apply Augmentation Button
        apply_btn = QPushButton("Apply Augmentations")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self.apply_augmentations)
        layout.addWidget(apply_btn)

        # Save Augmented Dataset Button
        save_btn = QPushButton("Save Augmented Dataset")
        save_btn.clicked.connect(self.save_augmented_dataset)
        layout.addWidget(save_btn)

        return panel

    # ---- AWGN page ----
    def _create_awgn_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # AWGN Toggle
        awgn_header = QHBoxLayout()
        awgn_left = QVBoxLayout()
        awgn_title = QLabel("AWGN (Additive White Gaussian Noise)")
        awgn_title.setProperty("class", "section-title")
        awgn_desc = QLabel("Add white Gaussian noise")
        awgn_desc.setProperty("class", "section-subtitle")
        awgn_left.addWidget(awgn_title)
        awgn_left.addWidget(awgn_desc)
        awgn_header.addLayout(awgn_left)
        awgn_header.addStretch()
        self.awgn_toggle = ToggleSwitch()
        self.awgn_toggle.setChecked(self.awgn_enabled)
        self.awgn_toggle.toggled.connect(lambda checked: setattr(self, 'awgn_enabled', checked))
        awgn_header.addWidget(self.awgn_toggle)
        layout.addLayout(awgn_header)

        # SNR Slider
        snr_slider_layout = self.create_slider_control(
            "SNR (dB)", self.snr_db, "dB", -10, 40, "snr_db"
        )
        layout.addLayout(snr_slider_layout)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Amplitude & Phase Shift
        amp_phase_header = QHBoxLayout()
        amp_phase_left = QVBoxLayout()
        amp_phase_title = QLabel("Amplitude & Phase Shift")
        amp_phase_title.setProperty("class", "section-title")
        amp_phase_desc = QLabel("Scale amplitude and rotate phase")
        amp_phase_desc.setProperty("class", "section-subtitle")
        amp_phase_left.addWidget(amp_phase_title)
        amp_phase_left.addWidget(amp_phase_desc)
        amp_phase_header.addLayout(amp_phase_left)
        amp_phase_header.addStretch()
        self.amp_phase_toggle = ToggleSwitch()
        self.amp_phase_toggle.setChecked(self.amp_phase_enabled)
        self.amp_phase_toggle.toggled.connect(lambda checked: setattr(self, 'amp_phase_enabled', checked))
        amp_phase_header.addWidget(self.amp_phase_toggle)
        layout.addLayout(amp_phase_header)

        # Amplitude Control
        amp_label = QLabel("Amplitude Scaling")
        layout.addWidget(amp_label)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 5.0)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setValue(self.amplitude)
        self.amplitude_spin.valueChanged.connect(lambda v: setattr(self, 'amplitude', v))
        layout.addWidget(self.amplitude_spin)

        # Phase Control
        phase_label = QLabel("Phase Shift (degrees)")
        layout.addWidget(phase_label)
        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-180.0, 180.0)
        self.phase_spin.setSingleStep(5.0)
        self.phase_spin.setValue(self.phase_deg)
        self.phase_spin.valueChanged.connect(lambda v: setattr(self, 'phase_deg', v))
        layout.addWidget(self.phase_spin)

        return page

    # ---- Stochastic TDL page ----
    def _create_stochastic_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # --- Header ---
        stoch_title = QLabel("TDL Fading Channel")
        stoch_title.setProperty("class", "section-title")
        stoch_desc = QLabel("Stochastic time-varying multipath channel (requires TensorFlow + Sionna)")
        stoch_desc.setProperty("class", "section-subtitle")
        layout.addWidget(stoch_title)
        layout.addWidget(stoch_desc)

        # --- Channel Parameters (grid) ---
        ch_grid = QGridLayout()
        ch_grid.setHorizontalSpacing(12)
        ch_grid.setVerticalSpacing(8)
        ch_grid.setColumnStretch(1, 1)

        ch_grid.addWidget(QLabel("TDL Profile"), 0, 0)
        self.tdl_profile_combo = QComboBox()
        self.tdl_profile_combo.addItems(["A", "B", "C", "D", "E"])
        self.tdl_profile_combo.setCurrentText(self.tdl_profile)
        self.tdl_profile_combo.currentTextChanged.connect(
            lambda v: setattr(self, 'tdl_profile', v))
        ch_grid.addWidget(self.tdl_profile_combo, 0, 1)

        ch_grid.addWidget(QLabel("Delay Spread (ns)"), 1, 0)
        self.delay_spread_spin = QDoubleSpinBox()
        self.delay_spread_spin.setRange(1.0, 1000.0)
        self.delay_spread_spin.setSingleStep(10.0)
        self.delay_spread_spin.setValue(self.delay_spread_ns)
        self.delay_spread_spin.valueChanged.connect(
            lambda v: setattr(self, 'delay_spread_ns', v))
        ch_grid.addWidget(self.delay_spread_spin, 1, 1)

        ch_grid.addWidget(QLabel("Max Doppler Speed (m/s)"), 2, 0)
        self.doppler_spin = QDoubleSpinBox()
        self.doppler_spin.setRange(0.0, 50.0)
        self.doppler_spin.setSingleStep(0.5)
        self.doppler_spin.setValue(self.max_doppler_speed)
        self.doppler_spin.valueChanged.connect(
            lambda v: setattr(self, 'max_doppler_speed', v))
        ch_grid.addWidget(self.doppler_spin, 2, 1)

        layout.addLayout(ch_grid)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Noise & Reproducibility ---
        stoch_snr_layout = self.create_slider_control(
            "SNR (dB)", self.stoch_snr_db, "dB", -10, 40, "stoch_snr_db"
        )
        layout.addLayout(stoch_snr_layout)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random Seed"))
        seed_row.addStretch()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(self.stoch_seed)
        self.seed_spin.setFixedWidth(100)
        self.seed_spin.valueChanged.connect(
            lambda v: setattr(self, 'stoch_seed', v))
        seed_row.addWidget(self.seed_spin)
        layout.addLayout(seed_row)

        # --- Separator ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # --- Dataset Info (read-only grid) ---
        info_title = QLabel("Dataset Info")
        info_title.setProperty("class", "section-title")
        layout.addWidget(info_title)

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(6)
        info_grid.setColumnStretch(1, 1)

        fc_label = QLabel("Carrier Frequency")
        fc_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(fc_label, 0, 0)
        self.carrier_freq_display = QLabel("--")
        self.carrier_freq_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_grid.addWidget(self.carrier_freq_display, 0, 1)

        sr_label = QLabel("Sample Rate")
        sr_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(sr_label, 1, 0)
        self.sample_rate_display = QLabel("--")
        self.sample_rate_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_grid.addWidget(self.sample_rate_display, 1, 1)

        out_label = QLabel("Output Samples")
        out_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(out_label, 2, 0)
        self.output_samples_spin = QSpinBox()
        self.output_samples_spin.setRange(128, 65536)
        self.output_samples_spin.setSingleStep(256)
        self.output_samples_spin.setValue(self.output_num_samples)
        self.output_samples_spin.valueChanged.connect(
            lambda v: setattr(self, 'output_num_samples', v))
        info_grid.addWidget(self.output_samples_spin, 2, 1)

        layout.addLayout(info_grid)

        return page

    def _on_subtab_changed(self, idx):
        self.active_subtab = idx
        self.subtab_stack.setCurrentIndex(idx)

    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        comparison_card = QFrame()
        comparison_card.setObjectName("card")
        comparison_layout = QVBoxLayout(comparison_card)
        comparison_layout.setContentsMargins(24, 24, 24, 24)

        comparison_title = QLabel("Clean vs Augmented Signal")
        comparison_title.setProperty("class", "card-title")
        comparison_layout.addWidget(comparison_title)

        from widgets.comparison_widget import ComparisonWidget
        self.comparison_plot = ComparisonWidget()
        comparison_layout.addWidget(self.comparison_plot)

        layout.addWidget(comparison_card)

        return panel

    def create_slider_control(self, label, value, unit, min_val, max_val, attr_name):
        """Create a slider control with label and value display"""
        container = QVBoxLayout()
        container.setSpacing(4)

        header = QHBoxLayout()
        label_widget = QLabel(label)
        value_label = QLabel(f"{value} {unit}")
        value_label.setProperty("class", "stat-value")
        header.addWidget(label_widget)
        header.addStretch()
        header.addWidget(value_label)
        container.addLayout(header)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(value))

        def update_value(v):
            value_label.setText(f"{v} {unit}")
            setattr(self, attr_name, float(v))

        slider.valueChanged.connect(update_value)
        container.addWidget(slider)

        return container

    def on_dataset_changed(self, name: str):
        if name and name != "No datasets loaded":
            self.dataset_manager.set_active(name)
            self.display_original_signal()
            self._update_stochastic_metadata()

    def _update_stochastic_metadata(self):
        """Auto-populate read-only carrier freq / sample rate from dataset metadata."""
        dataset = self.dataset_manager.get_active()
        if dataset is None:
            self.carrier_freq_display.setText("--")
            self.sample_rate_display.setText("--")
            return

        fs = dataset['fs']
        metadata = dataset.get('metadata', {})
        fc = metadata.get('fc', None)

        self.sample_rate_display.setText(f"{fs:,.0f}")
        if fc is not None:
            self.carrier_freq_display.setText(f"{float(fc):,.0f}")
        else:
            self.carrier_freq_display.setText("N/A (no fc in metadata)")

        # Default output_num_samples to signal length
        sig_len = len(dataset['signal'])
        self.output_num_samples = sig_len
        self.output_samples_spin.setValue(sig_len)

    def upload_dataset(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "NumPy Files (*.npy);;All Files (*)")

        if not filepath:
            return

        try:
            signal = np.load(filepath)

            from pathlib import Path
            name = Path(filepath).stem

            fs = 8e6
            metadata = {
                'source': 'uploaded',
                'filepath': filepath
            }

            self.dataset_manager.add_dataset(name, signal, fs, metadata)
            self.refresh_dataset_list()

            print(f"Uploaded dataset: {name}")

        except Exception as e:
            print(f"Failed to upload dataset: {e}")

    def refresh_dataset_list(self):
        self.dataset_combo.clear()
        datasets = self.dataset_manager.list_datasets()

        if datasets:
            self.dataset_combo.addItems(datasets)
            active = self.dataset_manager.active_dataset
            if active and active in datasets:
                self.dataset_combo.setCurrentText(active)
        else:
            self.dataset_combo.addItem("No datasets loaded")

        print(f"Dataset list refreshed: {datasets}")

    def display_original_signal(self):
        dataset = self.dataset_manager.get_active()
        if dataset is None:
            print("No active dataset to display")
            return

        self.clean_signal = dataset['signal']
        self.clean_fs = dataset['fs']
        self.clean_metadata = dataset.get('metadata', {})

    # ---- Config builder (CLIP_datagen schema) ----
    def _build_stochastic_config(self):
        dataset = self.dataset_manager.get_active()
        metadata = dataset.get('metadata', {}) if dataset else {}
        fs = dataset['fs'] if dataset else 8e6
        return {
            "seed": self.stoch_seed,
            "waveform": {
                "path": None,
                "format": "npy",
                "mat_key": None,
                "sample_rate_hz": fs,
                "normalize_rms": None,
                "iq_layout": "auto",
            },
            "augmentation": {
                "output_num_samples": self.output_num_samples,
                "pad_mode": "repeat",
                "trim_mode": "first",
            },
            "channel": {
                "type": "tdl",
                "profile": self.tdl_profile,
                "delay_spread_s": self.delay_spread_ns * 1e-9,
                "carrier_frequency_hz": float(metadata.get('fc', 1e6)),
                "sample_rate_hz": fs,
                "min_speed_mps": 0.0,
                "max_speed_mps": self.max_doppler_speed,
                "num_tx_ant": 1,
                "num_rx_ant": 1,
                "normalize_channel": False,
                "maximum_delay_spread_s": 3e-6,
                "l_min": None,
                "l_max": None,
            },
            "noise": {
                "snr_db": float(self.stoch_snr_db),
            },
        }

    def apply_augmentations(self):
        if not hasattr(self, 'clean_signal'):
            print("No dataset selected to augment.")
            return

        signal = self.clean_signal
        fs = self.clean_fs

        if self.active_subtab == 1:
            # Stochastic TDL path
            try:
                config = self._build_stochastic_config()
                block = StochasticTDLAugmentation(config)
                augmented_signal = block.apply(signal, fs)
            except ImportError as e:
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    f"Stochastic TDL augmentation requires TensorFlow and Sionna.\n\n{e}"
                )
                return
            except Exception as e:
                QMessageBox.critical(
                    self, "Augmentation Error",
                    f"Stochastic augmentation failed:\n\n{e}"
                )
                return
        else:
            # AWGN path
            pipeline = AugmentationPipeline()
            if self.awgn_enabled:
                pipeline.add(AWGNAugmentation(snr_db=self.snr_db))
            if self.amp_phase_enabled:
                phase_rad = np.deg2rad(self.phase_deg)
                pipeline.add(ScalarAmplitudeAndPhaseShift(amplitude=self.amplitude, phi=phase_rad))
            augmented_signal = pipeline.apply(signal, fs)

        # Get metadata for constellation plot
        metadata = self.clean_metadata
        fc = metadata.get('fc', None)
        Tsymb = metadata.get('Tsymb', None)
        modulation = metadata.get('modulation', None)
        M = metadata.get('M', None)
        alpha = metadata.get('alpha', 0.35)
        span = metadata.get('span', 8)
        pulse_shape = metadata.get('pulse_shape', 'rrc')
        nsymb = metadata.get('Nsymb', None)

        sps = None
        if Tsymb is not None:
            sps = int(fs * Tsymb)

        # Display comparison
        self.comparison_plot.plot_comparison(
            clean_signal=signal,
            augmented_signal=augmented_signal,
            fs=fs,
            fc=fc,
            sps=sps,
            modulation=modulation,
            M=M,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            nsymb=nsymb,
        )

        # Store augmented signal
        self.last_augmented_signal = augmented_signal
        self.last_augmented_fs = fs

    def save_augmented_dataset(self):
        if not hasattr(self, 'last_augmented_signal'):
            print("No augmented signal to save")
            return

        active_name = self.dataset_manager.active_dataset
        if not active_name:
            active_name = "dataset"

        aug_name = f"{active_name}_augmented"
        counter = 1
        while aug_name in self.dataset_manager.datasets:
            aug_name = f"{active_name}_augmented_{counter}"
            counter += 1

        # Get original metadata
        original_dataset = self.dataset_manager.get_active()
        metadata = original_dataset['metadata'].copy() if original_dataset else {}

        # Build structured augmentation metadata
        metadata['augmented'] = True
        metadata['base_dataset'] = active_name

        if self.active_subtab == 1:
            # Stochastic: save full CLIP_datagen config
            config = self._build_stochastic_config()
            metadata['augmentation_type'] = 'stochastic_tdl'
            metadata['augmentation_config'] = config
        else:
            # AWGN: structured params
            metadata['augmentation_type'] = 'awgn'
            metadata['augmentation_config'] = {
                'awgn': {
                    'enabled': self.awgn_enabled,
                    'snr_db': self.snr_db,
                },
                'amp_phase': {
                    'enabled': self.amp_phase_enabled,
                    'amplitude': self.amplitude,
                    'phase_deg': self.phase_deg,
                },
            }

        # Store in DatasetManager in-memory
        self.dataset_manager.add_dataset(
            aug_name,
            self.last_augmented_signal,
            self.last_augmented_fs,
            metadata
        )
        self.refresh_dataset_list()
        print(f"Saved augmented dataset in memory: {aug_name}")

        # Prompt for disk export
        export_dir = QFileDialog.getExistingDirectory(self, "Export Augmented Dataset To...")
        if export_dir:
            npy_path = os.path.join(export_dir, f"{aug_name}.npy")
            json_path = os.path.join(export_dir, f"{aug_name}.json")

            np.save(npy_path, self.last_augmented_signal)

            # Build config JSON for disk
            if self.active_subtab == 1:
                disk_config = self._build_stochastic_config()
                # Point waveform.path to the companion .npy
                disk_config["waveform"]["path"] = f"{aug_name}.npy"
            else:
                disk_config = metadata['augmentation_config']

            with open(json_path, 'w') as f:
                json.dump(disk_config, f, indent=2, default=str)

            print(f"Exported to disk: {npy_path}, {json_path}")
