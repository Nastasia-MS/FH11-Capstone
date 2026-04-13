from datetime import datetime

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSlider, QGridLayout, QComboBox,
                               QDoubleSpinBox, QSpinBox, QStackedWidget, QMessageBox,
                               QButtonGroup, QCheckBox, QScrollArea, QProgressDialog)
from PySide6.QtCore import Qt, QSettings

from PySide6.QtGui import QPalette, QColor

from mixedsignal_gui.widgets.toggle_switch import ToggleSwitch
from mixedsignal_gui.widgets.noise_spectrum import NoiseSpectrumWidget
from mixedsignal_gui.widgets.waveform_plots import PlottingWidget
from mixedsignal_gui.widgets.range_spinbox import RangeSpinBox
from mixedsignal_gui.backend.augmentation import (AugmentationPipeline, AWGNAugmentation,
                                  ScalarAmplitudeAndPhaseShift, FrequencyShift,
                                  StochasticTDLAugmentation, SionnaRTAugmentation)
from mixedsignal_gui.backend.parameter_range import ParameterRange
from mixedsignal_gui.backend.bulk_augmentation import BulkAugmentationThread
from mixedsignal_gui.backend.dataset_manager import DatasetManager
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
        self.freq_shift_hz = 0.0
        self.awgn_enabled = True
        self.amp_phase_enabled = False
        self.freq_shift_enabled = False

        # Stochastic TDL params
        self.active_subtab = 0  # 0=AWGN, 1=Stochastic, 2=RT
        self.tdl_profile = "A"
        self.delay_spread_ns = 100.0
        self.stoch_snr_db = 20.0
        self.stoch_seed = 42
        self.output_num_samples = 1024

        # RT state
        self.rt_last_taps = None
        self.sionna_widget = None  # lazy-created

        self._active_entry = None   # currently selected metadata dict

        self.setup_ui()
        self.refresh_dataset_list()

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

        self.rt_pill = QPushButton("Ray Tracing")
        self.rt_pill.setObjectName("subtabPill")
        self.rt_pill.setCheckable(True)

        self.subtab_group.addButton(self.awgn_pill, 0)
        self.subtab_group.addButton(self.stoch_pill, 1)
        self.subtab_group.addButton(self.rt_pill, 2)

        pill_layout.addWidget(self.awgn_pill)
        pill_layout.addWidget(self.stoch_pill)
        pill_layout.addWidget(self.rt_pill)
        pill_layout.addStretch()
        layout.addLayout(pill_layout)

        # --- Stacked widget ---
        self.subtab_stack = QStackedWidget()
        self.subtab_stack.addWidget(self._create_awgn_page())
        self.subtab_stack.addWidget(self._create_stochastic_page())
        self.subtab_stack.addWidget(self._create_rt_page())
        layout.addWidget(self.subtab_stack, 1)  # stretch factor 1 → fills all space above buttons

        self.subtab_group.idClicked.connect(self._on_subtab_changed)

        # RT-only: Compute Paths button + status (hidden for AWGN/Stochastic)
        self.rt_compute_btn = QPushButton("Compute Paths")
        self.rt_compute_btn.setObjectName("primaryButton")
        self.rt_compute_btn.clicked.connect(self._rt_compute_paths)
        self.rt_compute_btn.setVisible(False)
        layout.addWidget(self.rt_compute_btn)

        self.rt_status_label = QLabel("No scene loaded")
        self.rt_status_label.setWordWrap(True)
        self.rt_status_label.setProperty("class", "section-subtitle")
        self.rt_status_label.setVisible(False)
        layout.addWidget(self.rt_status_label)

        # Apply Augmentation Button
        apply_btn = QPushButton("Apply Augmentations")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self.apply_augmentations)
        layout.addWidget(apply_btn)

        # Bulk Apply Button
        bulk_btn = QPushButton("Apply to a Dataset Folder")
        bulk_btn.clicked.connect(self.apply_to_all_in_folder)
        layout.addWidget(bulk_btn)

        return panel

    @staticmethod
    def _make_scroll_page():
        """Create a scroll-area-wrapped page with white background matching QFrame#card.

        Returns (page, layout) where *page* goes into the QStackedWidget and
        *layout* is the QVBoxLayout to add controls into.
        """
        _white = QColor(255, 255, 255)
        _dark = QColor(0x1f, 0x29, 0x37)  # #1f2937 — matches stylesheet text

        def _apply_palette(w):
            pal = w.palette()
            pal.setColor(QPalette.Window, _white)
            pal.setColor(QPalette.Base, _white)
            pal.setColor(QPalette.WindowText, _dark)
            pal.setColor(QPalette.Text, _dark)
            w.setPalette(pal)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        _apply_palette(scroll)
        _apply_palette(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        scroll.setWidget(container)
        # Re-apply after setWidget since Qt may recreate the viewport
        _apply_palette(scroll.viewport())

        page_layout.addWidget(scroll)
        return page, layout

    # ---- AWGN page ----
    def _create_awgn_page(self):
        page, layout = self._make_scroll_page()

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
        self.awgn_toggle.toggled.connect(lambda checked: (
            self.snr_spin.vary_checkbox.setEnabled(checked),
            self.snr_spin.vary_checkbox.setChecked(False) if not checked else None,
        ))
        awgn_header.addWidget(self.awgn_toggle)
        layout.addLayout(awgn_header)

        # SNR (RangeSpinBox)
        self.snr_spin = RangeSpinBox()
        snr_header = QHBoxLayout()
        snr_header.addWidget(QLabel("SNR (dB)"))
        snr_header.addStretch()
        snr_header.addWidget(self.snr_spin.vary_checkbox)
        layout.addLayout(snr_header)
        self.snr_spin.setRange(-10.0, 40.0)
        self.snr_spin.setSingleStep(1.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setValue(self.snr_db)
        self.snr_spin.valueChanged.connect(lambda v: setattr(self, 'snr_db', v))
        layout.addWidget(self.snr_spin)

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
        self.amp_phase_toggle.toggled.connect(lambda checked: [
            self.amplitude_spin.vary_checkbox.setEnabled(checked),
            self.phase_spin.vary_checkbox.setEnabled(checked),
            self.amplitude_spin.vary_checkbox.setChecked(False) if not checked else None,
            self.phase_spin.vary_checkbox.setChecked(False) if not checked else None,
        ])
        amp_phase_header.addWidget(self.amp_phase_toggle)
        layout.addLayout(amp_phase_header)

        # Amplitude Control
        self.amplitude_spin = RangeSpinBox()
        self.amplitude_spin.setRange(0.0, 5.0)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setDecimals(2)
        self.amplitude_spin.setValue(self.amplitude)
        self.amplitude_spin.valueChanged.connect(lambda v: setattr(self, 'amplitude', v))
        amp_header = QHBoxLayout()
        amp_header.addWidget(QLabel("Amplitude Scaling"))
        amp_header.addStretch()
        amp_header.addWidget(self.amplitude_spin.vary_checkbox)
        layout.addLayout(amp_header)
        layout.addWidget(self.amplitude_spin)

        # Phase Control
        phase_header = QHBoxLayout()
        phase_header.addWidget(QLabel("Phase Shift (degrees)"))
        phase_header.addStretch()
        self.phase_spin = RangeSpinBox()
        phase_header.addWidget(self.phase_spin.vary_checkbox)
        layout.addLayout(phase_header)
        self.phase_spin.setRange(-180.0, 180.0)
        self.phase_spin.setSingleStep(5.0)
        self.phase_spin.setDecimals(1)
        self.phase_spin.setSuffix("°")
        self.phase_spin.setValue(self.phase_deg)
        self.phase_spin.valueChanged.connect(lambda v: setattr(self, 'phase_deg', v))
        layout.addWidget(self.phase_spin)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # Frequency Shift (CFO)
        freq_shift_header = QHBoxLayout()
        freq_shift_left = QVBoxLayout()
        freq_shift_title = QLabel("Frequency Offset (CFO)")
        freq_shift_title.setProperty("class", "section-title")
        freq_shift_desc = QLabel("Simulate carrier frequency offset from oscillator mismatch")
        freq_shift_desc.setProperty("class", "section-subtitle")
        freq_shift_left.addWidget(freq_shift_title)
        freq_shift_left.addWidget(freq_shift_desc)
        freq_shift_header.addLayout(freq_shift_left)
        freq_shift_header.addStretch()
        self.freq_shift_toggle = ToggleSwitch()
        self.freq_shift_toggle.setChecked(self.freq_shift_enabled)
        self.freq_shift_toggle.toggled.connect(lambda checked: setattr(self, 'freq_shift_enabled', checked))
        self.freq_shift_toggle.toggled.connect(lambda checked: (
            self.freq_shift_spin.vary_checkbox.setEnabled(checked),
            self.freq_shift_spin.vary_checkbox.setChecked(False) if not checked else None,
        ))
        freq_shift_header.addWidget(self.freq_shift_toggle)
        layout.addLayout(freq_shift_header)
        
        freq_shift_label = QHBoxLayout()
        freq_shift_label.addWidget(QLabel("Frequency Offset (Hz)"))
        freq_shift_label.addStretch()
        self.freq_shift_spin = RangeSpinBox()
        freq_shift_label.addWidget(self.freq_shift_spin.vary_checkbox)
        layout.addLayout(freq_shift_label)
        self.freq_shift_spin.setRange(-100e6, 100e6)
        self.freq_shift_spin.setSingleStep(1000.0)
        self.freq_shift_spin.setDecimals(0)
        self.freq_shift_spin.setSuffix(" Hz")
        self.freq_shift_spin.setValue(self.freq_shift_hz)
        self.freq_shift_spin.valueChanged.connect(lambda v: setattr(self, 'freq_shift_hz', v))
        layout.addWidget(self.freq_shift_spin)

        # Sync initial enabled state for vary checkboxes
        self.snr_spin.vary_checkbox.setEnabled(self.awgn_enabled)
        self.amplitude_spin.vary_checkbox.setEnabled(self.amp_phase_enabled)
        self.phase_spin.vary_checkbox.setEnabled(self.amp_phase_enabled)
        self.freq_shift_spin.vary_checkbox.setEnabled(self.freq_shift_enabled)

        return page

    # ---- Stochastic TDL page ----
    def _create_stochastic_page(self):
        page, layout = self._make_scroll_page()

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

        self.delay_spread_spin = RangeSpinBox()
        self.delay_spread_spin.setRange(1.0, 1000.0)
        self.delay_spread_spin.setSingleStep(10.0)
        self.delay_spread_spin.setDecimals(1)
        self.delay_spread_spin.setSuffix(" ns")
        self.delay_spread_spin.setValue(self.delay_spread_ns)
        self.delay_spread_spin.valueChanged.connect(
            lambda v: setattr(self, 'delay_spread_ns', v))
        delay_spread_label_widget = QWidget()
        delay_spread_label_lay = QHBoxLayout(delay_spread_label_widget)
        delay_spread_label_lay.setContentsMargins(0, 0, 0, 0)
        delay_spread_label_lay.addWidget(QLabel("Delay Spread (ns)"))
        delay_spread_label_lay.addWidget(self.delay_spread_spin.vary_checkbox)
        ch_grid.addWidget(delay_spread_label_widget, 1, 0)
        ch_grid.addWidget(self.delay_spread_spin, 1, 1)

        layout.addLayout(ch_grid)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Noise & Reproducibility ---
        stoch_snr_header = QHBoxLayout()
        stoch_snr_header.addWidget(QLabel("SNR (dB)"))
        stoch_snr_header.addStretch()
        self.stoch_snr_spin = RangeSpinBox()
        stoch_snr_header.addWidget(self.stoch_snr_spin.vary_checkbox)
        layout.addLayout(stoch_snr_header)
        self.stoch_snr_spin.setRange(-10.0, 40.0)
        self.stoch_snr_spin.setSingleStep(1.0)
        self.stoch_snr_spin.setDecimals(1)
        self.stoch_snr_spin.setSuffix(" dB")
        self.stoch_snr_spin.setValue(self.stoch_snr_db)
        self.stoch_snr_spin.valueChanged.connect(lambda v: setattr(self, 'stoch_snr_db', v))
        layout.addWidget(self.stoch_snr_spin)

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

        fc_label = QLabel("Carrier Frequency (MHz)")
        fc_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(fc_label, 0, 0)
        self.carrier_freq_display = QLabel("--")
        self.carrier_freq_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_grid.addWidget(self.carrier_freq_display, 0, 1)

        sr_label = QLabel("Sample Rate (MHz)")
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

        # --- Multi-Channel Output ---
        sep_mc = QFrame()
        sep_mc.setFrameShape(QFrame.HLine)
        sep_mc.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep_mc)

        mc_title = QLabel("Multi-Channel Array Output")
        mc_title.setProperty("class", "section-title")
        layout.addWidget(mc_title)

        self.stoch_multi_channel_cb = QCheckBox("Enable Multi-Channel Output")
        self.stoch_multi_channel_cb.setToolTip(
            "Return all RX antenna outputs instead of selecting a single one"
        )
        self.stoch_multi_channel_cb.toggled.connect(self._on_stoch_multi_channel_toggled)
        layout.addWidget(self.stoch_multi_channel_cb)

        stoch_mc_grid = QGridLayout()
        stoch_mc_grid.setHorizontalSpacing(12)
        stoch_mc_grid.setVerticalSpacing(8)
        stoch_mc_grid.setColumnStretch(1, 1)

        stoch_mc_grid.addWidget(QLabel("Num TX Antennas"), 0, 0)
        self.stoch_num_tx_ant_spin = QSpinBox()
        self.stoch_num_tx_ant_spin.setRange(1, 64)
        self.stoch_num_tx_ant_spin.setValue(1)
        self.stoch_num_tx_ant_spin.setEnabled(False)
        stoch_mc_grid.addWidget(self.stoch_num_tx_ant_spin, 0, 1)

        stoch_mc_grid.addWidget(QLabel("Num RX Antennas"), 1, 0)
        self.stoch_num_rx_ant_spin = QSpinBox()
        self.stoch_num_rx_ant_spin.setRange(1, 64)
        self.stoch_num_rx_ant_spin.setValue(1)
        self.stoch_num_rx_ant_spin.setEnabled(False)
        stoch_mc_grid.addWidget(self.stoch_num_rx_ant_spin, 1, 1)

        layout.addLayout(stoch_mc_grid)

        save_mode_row = QHBoxLayout()
        save_mode_row.addWidget(QLabel("Save Mode"))
        self.stoch_save_mode_combo = QComboBox()
        self.stoch_save_mode_combo.addItems([
            "Single Antenna",
            "All Antennas (Separate Files)",
            "All Antennas (Stacked)",
        ])
        self.stoch_save_mode_combo.setEnabled(False)
        save_mode_row.addWidget(self.stoch_save_mode_combo)
        layout.addLayout(save_mode_row)

        self.stoch_output_shape_label = QLabel("")
        self.stoch_output_shape_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.stoch_output_shape_label)

        return page

    # ---- Ray Tracing page (full controls) ----
    def _create_rt_page(self):
        """Create the RT page with all controls, styled to match AWGN/Stochastic pages."""
        page, layout = self._make_scroll_page()

        # --- Header ---
        rt_title = QLabel("Sionna Ray Tracing Channel")
        rt_title.setProperty("class", "section-title")
        rt_desc = QLabel("Deterministic channel from 3D scene ray tracing")
        rt_desc.setProperty("class", "section-subtitle")
        layout.addWidget(rt_title)
        layout.addWidget(rt_desc)

        # --- Scene ---
        self.rt_scene_label = QLabel("No scene loaded")
        self.rt_scene_label.setWordWrap(True)
        self.rt_scene_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.rt_scene_label)

        self.rt_load_scene_btn = QPushButton("Load Scene...")
        self.rt_load_scene_btn.clicked.connect(self._rt_load_scene)
        layout.addWidget(self.rt_load_scene_btn)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Simulation Parameters (grid) ---
        param_title = QLabel("Simulation Parameters")
        param_title.setProperty("class", "section-title")
        layout.addWidget(param_title)

        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(12)
        param_grid.setVerticalSpacing(8)
        param_grid.setColumnStretch(1, 1)

        fc_label = QLabel("Center Frequency (MHz)")
        fc_label.setProperty("class", "section-subtitle")
        param_grid.addWidget(fc_label, 0, 0)
        self.rt_freq_display = QLabel("--")
        self.rt_freq_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_grid.addWidget(self.rt_freq_display, 0, 1)

        sr_label = QLabel("Sample Rate (MHz)")
        sr_label.setProperty("class", "section-subtitle")
        param_grid.addWidget(sr_label, 1, 0)
        self.rt_sample_rate_display = QLabel("--")
        self.rt_sample_rate_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_grid.addWidget(self.rt_sample_rate_display, 1, 1)

        param_grid.addWidget(QLabel("Max Depth"), 2, 0)
        self.rt_depth_spin = QSpinBox()
        self.rt_depth_spin.setRange(1, 20)
        self.rt_depth_spin.setValue(3)
        self.rt_depth_spin.valueChanged.connect(self._rt_push_max_depth)
        param_grid.addWidget(self.rt_depth_spin, 2, 1)

        param_grid.addWidget(QLabel("Num Samples"), 3, 0)
        self.rt_samples_spin = QSpinBox()
        self.rt_samples_spin.setRange(1000, 10_000_000)
        self.rt_samples_spin.setSingleStep(100_000)
        self.rt_samples_spin.setValue(1_000_000)
        self.rt_samples_spin.valueChanged.connect(self._rt_push_num_samples)
        param_grid.addWidget(self.rt_samples_spin, 3, 1)

        layout.addLayout(param_grid)

        # --- Separator ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # --- Transceivers ---
        trx_title = QLabel("Transceivers")
        trx_title.setProperty("class", "section-title")
        layout.addWidget(trx_title)

        # TX Position
        layout.addWidget(QLabel("TX Position"))
        self.rt_tx_x = QDoubleSpinBox()
        self.rt_tx_y = QDoubleSpinBox()
        self.rt_tx_z = QDoubleSpinBox()
        for spin, default in [(self.rt_tx_x, 0.0), (self.rt_tx_y, 0.0), (self.rt_tx_z, 10.0)]:
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setValue(default)
            spin.valueChanged.connect(self._rt_push_tx_position)
        tx_pos_row = QHBoxLayout()
        tx_pos_row.setSpacing(4)
        tx_pos_row.addWidget(QLabel("X:"))
        tx_pos_row.addWidget(self.rt_tx_x)
        tx_pos_row.addWidget(QLabel("Y:"))
        tx_pos_row.addWidget(self.rt_tx_y)
        tx_pos_row.addWidget(QLabel("Z:"))
        tx_pos_row.addWidget(self.rt_tx_z)
        layout.addLayout(tx_pos_row)

        self.rt_place_tx_btn = QPushButton("Place TX")
        self.rt_place_tx_btn.setCheckable(True)
        self.rt_place_tx_btn.toggled.connect(self._rt_on_place_tx)
        layout.addWidget(self.rt_place_tx_btn)

        # RX Position
        layout.addWidget(QLabel("RX Position"))
        self.rt_rx_x = QDoubleSpinBox()
        self.rt_rx_y = QDoubleSpinBox()
        self.rt_rx_z = QDoubleSpinBox()
        for spin, default in [(self.rt_rx_x, 50.0), (self.rt_rx_y, 0.0), (self.rt_rx_z, 1.5)]:
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setValue(default)
            spin.valueChanged.connect(self._rt_push_rx_position)
        rx_pos_row = QHBoxLayout()
        rx_pos_row.setSpacing(4)
        rx_pos_row.addWidget(QLabel("X:"))
        rx_pos_row.addWidget(self.rt_rx_x)
        rx_pos_row.addWidget(QLabel("Y:"))
        rx_pos_row.addWidget(self.rt_rx_y)
        rx_pos_row.addWidget(QLabel("Z:"))
        rx_pos_row.addWidget(self.rt_rx_z)
        layout.addLayout(rx_pos_row)

        self.rt_place_rx_btn = QPushButton("Place RX")
        self.rt_place_rx_btn.setCheckable(True)
        self.rt_place_rx_btn.toggled.connect(self._rt_on_place_rx)
        layout.addWidget(self.rt_place_rx_btn)

        # --- Separator ---
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep3)

        # --- Antenna Arrays (grid) ---
        ant_title = QLabel("Antenna Arrays")
        ant_title.setProperty("class", "section-title")
        layout.addWidget(ant_title)

        ant_grid = QGridLayout()
        ant_grid.setHorizontalSpacing(12)
        ant_grid.setVerticalSpacing(8)
        ant_grid.setColumnStretch(1, 1)

        ant_grid.addWidget(QLabel("TX Rows"), 0, 0)
        self.rt_tx_ant_rows = QSpinBox()
        self.rt_tx_ant_rows.setRange(1, 16)
        self.rt_tx_ant_rows.setValue(6)
        self.rt_tx_ant_rows.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_tx_ant_rows, 0, 1)

        ant_grid.addWidget(QLabel("TX Cols"), 1, 0)
        self.rt_tx_ant_cols = QSpinBox()
        self.rt_tx_ant_cols.setRange(1, 16)
        self.rt_tx_ant_cols.setValue(6)
        self.rt_tx_ant_cols.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_tx_ant_cols, 1, 1)

        ant_grid.addWidget(QLabel("TX Pattern"), 2, 0)
        self.rt_tx_ant_pattern = QComboBox()
        self.rt_tx_ant_pattern.addItems(["tr38901", "iso"])
        self.rt_tx_ant_pattern.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_tx_ant_pattern, 2, 1)

        ant_grid.addWidget(QLabel("TX Polarization"), 3, 0)
        self.rt_tx_ant_pol = QComboBox()
        self.rt_tx_ant_pol.addItems(["V", "H", "cross"])
        self.rt_tx_ant_pol.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_tx_ant_pol, 3, 1)

        ant_grid.addWidget(QLabel("RX Rows"), 4, 0)
        self.rt_rx_ant_rows = QSpinBox()
        self.rt_rx_ant_rows.setRange(1, 16)
        self.rt_rx_ant_rows.setValue(6)
        self.rt_rx_ant_rows.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_rx_ant_rows, 4, 1)

        ant_grid.addWidget(QLabel("RX Cols"), 5, 0)
        self.rt_rx_ant_cols = QSpinBox()
        self.rt_rx_ant_cols.setRange(1, 16)
        self.rt_rx_ant_cols.setValue(6)
        self.rt_rx_ant_cols.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_rx_ant_cols, 5, 1)

        ant_grid.addWidget(QLabel("RX Pattern"), 6, 0)
        self.rt_rx_ant_pattern = QComboBox()
        self.rt_rx_ant_pattern.addItems(["iso", "tr38901"])
        self.rt_rx_ant_pattern.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_rx_ant_pattern, 6, 1)

        ant_grid.addWidget(QLabel("RX Polarization"), 7, 0)
        self.rt_rx_ant_pol = QComboBox()
        self.rt_rx_ant_pol.addItems(["V", "H", "cross"])
        self.rt_rx_ant_pol.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_rx_ant_pol, 7, 1)

        layout.addLayout(ant_grid)

        # --- Separator ---
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.HLine)
        sep4.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep4)

        # --- Path Solver ---
        solver_title = QLabel("Path Solver")
        solver_title.setProperty("class", "section-title")
        layout.addWidget(solver_title)

        solver_toggles = [
            ("Line of Sight",       "rt_cb_los"),
            ("Specular Reflection", "rt_cb_specular"),
            ("Diffuse Reflection",  "rt_cb_diffuse"),
            ("Refraction",          "rt_cb_refraction"),
            ("Synthetic Array",     "rt_cb_synthetic"),
        ]
        for label_text, attr_name in solver_toggles:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addStretch()
            toggle = ToggleSwitch()
            toggle.setChecked(True)
            toggle.toggled.connect(lambda _: self._rt_push_solver())
            setattr(self, attr_name, toggle)
            row.addWidget(toggle)
            layout.addLayout(row)

        # --- Separator ---
        sep5 = QFrame()
        sep5.setFrameShape(QFrame.HLine)
        sep5.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep5)

        # --- Channel Parameters (grid) ---
        ch_title = QLabel("Channel Parameters")
        ch_title.setProperty("class", "section-title")
        layout.addWidget(ch_title)

        ch_grid = QGridLayout()
        ch_grid.setHorizontalSpacing(12)
        ch_grid.setVerticalSpacing(8)
        ch_grid.setColumnStretch(1, 1)

        ch_grid.addWidget(QLabel("Bandwidth (MHz)"), 0, 0)
        self.rt_bandwidth_spin = QDoubleSpinBox()
        self.rt_bandwidth_spin.setRange(0.01, 1000.0)
        self.rt_bandwidth_spin.setValue(5.0)
        self.rt_bandwidth_spin.setDecimals(2)
        ch_grid.addWidget(self.rt_bandwidth_spin, 0, 1)

        ch_grid.addWidget(QLabel("l_min"), 1, 0)
        self.rt_l_min_spin = QSpinBox()
        self.rt_l_min_spin.setRange(0, 10000)
        self.rt_l_min_spin.setValue(0)
        ch_grid.addWidget(self.rt_l_min_spin, 1, 1)

        ch_grid.addWidget(QLabel("l_max"), 2, 0)
        self.rt_l_max_spin = QSpinBox()
        self.rt_l_max_spin.setRange(1, 10000)
        self.rt_l_max_spin.setValue(200)
        ch_grid.addWidget(self.rt_l_max_spin, 2, 1)

        ch_grid.addWidget(QLabel("TX Power (dBm)"), 3, 0)
        self.rt_tx_power_spin = QDoubleSpinBox()
        self.rt_tx_power_spin.setRange(-30.0, 60.0)
        self.rt_tx_power_spin.setValue(30.0)
        self.rt_tx_power_spin.setDecimals(1)
        ch_grid.addWidget(self.rt_tx_power_spin, 3, 1)

        ch_grid.addWidget(QLabel("Noise Power (dBm)"), 4, 0)
        self.rt_noise_power_spin = QDoubleSpinBox()
        self.rt_noise_power_spin.setRange(-200.0, 0.0)
        self.rt_noise_power_spin.setValue(-108.0)
        self.rt_noise_power_spin.setDecimals(1)
        ch_grid.addWidget(self.rt_noise_power_spin, 4, 1)

        ch_grid.addWidget(QLabel("Waveform Length"), 5, 0)
        self.rt_waveform_length_spin = QSpinBox()
        self.rt_waveform_length_spin.setRange(64, 65536)
        self.rt_waveform_length_spin.setValue(1024)
        self.rt_waveform_length_spin.setSingleStep(256)
        ch_grid.addWidget(self.rt_waveform_length_spin, 5, 1)

        ch_grid.addWidget(QLabel("Waveform Padding"), 6, 0)
        self.rt_padding_combo = QComboBox()
        self.rt_padding_combo.addItems(["Zero Padded", "Tile / Repeat"])
        ch_grid.addWidget(self.rt_padding_combo, 6, 1)

        ch_grid.addWidget(QLabel("RX Antenna Index"), 7, 0)
        self.rt_rx_antenna_idx_spin = QSpinBox()
        self.rt_rx_antenna_idx_spin.setRange(0, 0)
        self.rt_rx_antenna_idx_spin.setValue(0)
        ch_grid.addWidget(self.rt_rx_antenna_idx_spin, 7, 1)

        layout.addLayout(ch_grid)

        # --- Multi-Channel Output ---
        sep_mc = QFrame()
        sep_mc.setFrameShape(QFrame.HLine)
        sep_mc.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep_mc)

        mc_title = QLabel("Multi-Channel Array Output")
        mc_title.setProperty("class", "section-title")
        layout.addWidget(mc_title)

        self.rt_multi_channel_cb = QCheckBox("Enable Multi-Channel Output")
        self.rt_multi_channel_cb.setToolTip(
            "Return all RX antenna outputs instead of selecting a single one"
        )
        self.rt_multi_channel_cb.toggled.connect(self._on_rt_multi_channel_toggled)
        layout.addWidget(self.rt_multi_channel_cb)

        save_mode_row = QHBoxLayout()
        save_mode_row.addWidget(QLabel("Save Mode"))
        self.rt_save_mode_combo = QComboBox()
        self.rt_save_mode_combo.addItems([
            "Single Antenna",
            "All Antennas (Separate Files)",
            "All Antennas (Stacked)",
        ])
        self.rt_save_mode_combo.setEnabled(False)
        save_mode_row.addWidget(self.rt_save_mode_combo)
        layout.addLayout(save_mode_row)

        self.rt_output_shape_label = QLabel("")
        self.rt_output_shape_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.rt_output_shape_label)

        return page

    # ---- RT control push methods ----

    def _rt_load_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Sionna Scene", "",
            "XML Files (*.xml);;All Files (*)",
        )
        if not path:
            return
        if not self._ensure_sionna_widget():
            return

        # Clear stale taps from a previous scene
        self.rt_last_taps = None

        # Re-enable compute button in case it was stuck from a prior error
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

        try:
            self.sionna_widget.load_scene(path)
            self.rt_scene_label.setText(path.rsplit("/", 1)[-1])
        except Exception as e:
            self.rt_status_label.setText(f"Error loading scene: {e}")
            print(f"Failed to load scene: {e}")

    def _rt_push_max_depth(self, val):
        if self.sionna_widget is not None:
            self.sionna_widget.set_max_depth(val)

    def _rt_push_num_samples(self, val):
        if self.sionna_widget is not None:
            self.sionna_widget.set_num_samples(val)

    def _rt_push_tx_position(self, _=None):
        if self.sionna_widget is not None:
            pos = [self.rt_tx_x.value(), self.rt_tx_y.value(), self.rt_tx_z.value()]
            self.sionna_widget.set_tx_position(pos)

    def _rt_push_rx_position(self, _=None):
        if self.sionna_widget is not None:
            pos = [self.rt_rx_x.value(), self.rt_rx_y.value(), self.rt_rx_z.value()]
            self.sionna_widget.set_rx_position(pos)

    def _rt_on_place_tx(self, on):
        if on:
            self.rt_place_rx_btn.blockSignals(True)
            self.rt_place_rx_btn.setChecked(False)
            self.rt_place_rx_btn.blockSignals(False)
            if self.sionna_widget is not None:
                self.sionna_widget.enter_placement_mode("tx")

    def _rt_on_place_rx(self, on):
        if on:
            self.rt_place_tx_btn.blockSignals(True)
            self.rt_place_tx_btn.setChecked(False)
            self.rt_place_tx_btn.blockSignals(False)
            if self.sionna_widget is not None:
                self.sionna_widget.enter_placement_mode("rx")

    def _rt_uncheck_placement_buttons(self):
        self.rt_place_tx_btn.blockSignals(True)
        self.rt_place_rx_btn.blockSignals(True)
        self.rt_place_tx_btn.setChecked(False)
        self.rt_place_rx_btn.setChecked(False)
        self.rt_place_tx_btn.blockSignals(False)
        self.rt_place_rx_btn.blockSignals(False)

    def _on_rt_multi_channel_toggled(self, checked):
        self.rt_rx_antenna_idx_spin.setEnabled(not checked)
        self.rt_save_mode_combo.setEnabled(checked)
        if not checked:
            self.rt_save_mode_combo.setCurrentIndex(0)
            self.rt_output_shape_label.setText("")

    def _on_stoch_multi_channel_toggled(self, checked):
        self.stoch_num_tx_ant_spin.setEnabled(checked)
        self.stoch_num_rx_ant_spin.setEnabled(checked)
        self.stoch_save_mode_combo.setEnabled(checked)
        if not checked:
            self.stoch_save_mode_combo.setCurrentIndex(0)
            self.stoch_output_shape_label.setText("")

    def _rt_push_antennas(self, _=None):
        if self.sionna_widget is None:
            return
        tx_cfg = {
            "num_rows": self.rt_tx_ant_rows.value(),
            "num_cols": self.rt_tx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.rt_tx_ant_pattern.currentText(),
            "polarization": self.rt_tx_ant_pol.currentText(),
            "polarization_model": "tr38901_2",
        }
        rx_cfg = {
            "num_rows": self.rt_rx_ant_rows.value(),
            "num_cols": self.rt_rx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.rt_rx_ant_pattern.currentText(),
            "polarization": self.rt_rx_ant_pol.currentText(),
            "polarization_model": "tr38901_2",
        }
        self.sionna_widget.set_antenna_arrays(tx_cfg, rx_cfg)
        # Update RX antenna index max
        max_idx = rx_cfg["num_rows"] * rx_cfg["num_cols"] - 1
        self.rt_rx_antenna_idx_spin.setMaximum(max(max_idx, 0))
        if self.rt_rx_antenna_idx_spin.value() > max_idx:
            self.rt_rx_antenna_idx_spin.setValue(0)

    def _rt_push_solver(self):
        if self.sionna_widget is None:
            return
        opts = {
            "line_of_sight": self.rt_cb_los.isChecked(),
            "specular_reflection": self.rt_cb_specular.isChecked(),
            "diffuse_reflection": self.rt_cb_diffuse.isChecked(),
            "refraction": self.rt_cb_refraction.isChecked(),
            "synthetic_array": self.rt_cb_synthetic.isChecked(),
        }
        self.sionna_widget.set_solver_options(opts)

    def _rt_get_channel_params_dict(self):
        # sample_rate_hz is intentionally omitted here — it is overlaid
        # from the active dataset's fs in _build_rt_config(), matching the
        # CLIP_Datagen pipeline where bandwidth and sample_rate are distinct.
        return {
            "bandwidth_hz": self.rt_bandwidth_spin.value() * 1e6,
            "l_min": self.rt_l_min_spin.value(),
            "l_max": self.rt_l_max_spin.value(),
            "tx_power_dbm": self.rt_tx_power_spin.value(),
            "noise_power_dbm": self.rt_noise_power_spin.value(),
            "waveform_length": self.rt_waveform_length_spin.value(),
            "zero_padding": self.rt_padding_combo.currentText(),
            "rx_antenna_index": self.rt_rx_antenna_idx_spin.value(),
        }

    def _rt_push_channel_params(self):
        if self.sionna_widget is not None:
            self.sionna_widget.set_channel_params(self._rt_get_channel_params_dict())

    def _rt_compute_paths(self):
        if not self._ensure_sionna_widget():
            return

        # Set engine frequency from dataset metadata before computing paths.
        entry = self._active_entry
        if entry:
            fc = entry.get('fc', None)
            if fc is not None:
                try:
                    self.sionna_widget.set_frequency_ghz(float(fc) / 1e9)
                except Exception as e:
                    self.rt_status_label.setText(
                        f"Warning: could not set frequency {float(fc)/1e6:.4g} MHz — {e}")

        # Push all current params before computing
        self._rt_push_channel_params()
        self.rt_compute_btn.setEnabled(False)
        self.rt_compute_btn.setText("Computing...")
        self.sionna_widget.compute_paths()

    # ---- RT viewport placement feedback ----

    def _on_rt_tx_placed(self, pos):
        """Viewport placed TX → update spinboxes."""
        self.rt_tx_x.blockSignals(True)
        self.rt_tx_y.blockSignals(True)
        self.rt_tx_z.blockSignals(True)
        self.rt_tx_x.setValue(pos[0])
        self.rt_tx_y.setValue(pos[1])
        self.rt_tx_z.setValue(pos[2])
        self.rt_tx_x.blockSignals(False)
        self.rt_tx_y.blockSignals(False)
        self.rt_tx_z.blockSignals(False)
        self._rt_uncheck_placement_buttons()

    def _on_rt_rx_placed(self, pos):
        """Viewport placed RX → update spinboxes."""
        self.rt_rx_x.blockSignals(True)
        self.rt_rx_y.blockSignals(True)
        self.rt_rx_z.blockSignals(True)
        self.rt_rx_x.setValue(pos[0])
        self.rt_rx_y.setValue(pos[1])
        self.rt_rx_z.setValue(pos[2])
        self.rt_rx_x.blockSignals(False)
        self.rt_rx_y.blockSignals(False)
        self.rt_rx_z.blockSignals(False)
        self._rt_uncheck_placement_buttons()

    # ---- subtab switching ----

    def _on_subtab_changed(self, idx):
        self.active_subtab = idx
        self.subtab_stack.setCurrentIndex(idx)

        is_rt = idx == 2
        self.rt_compute_btn.setVisible(is_rt)
        self.rt_status_label.setVisible(is_rt)

        if is_rt:
            # RT active: show viewport + comparison subtabs, hide plain comparison
            self.rt_right_container.setVisible(True)
            self.comparison_card.setVisible(False)
            self._ensure_sionna_widget()
        else:
            # AWGN / Stochastic: show plain comparison, hide RT container
            self.rt_right_container.setVisible(False)
            self.comparison_card.setVisible(True)

    # ---- Visualization (right) panel ----

    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── RT right container (hidden by default) ───────────
        self.rt_right_container = QWidget()
        rt_container_layout = QVBoxLayout(self.rt_right_container)
        rt_container_layout.setContentsMargins(0, 0, 0, 0)
        rt_container_layout.setSpacing(8)

        # Pill toggle: 3D Viewport | Signal Comparison
        rt_pill_layout = QHBoxLayout()
        rt_pill_layout.setSpacing(8)
        self.rt_view_group = QButtonGroup(self)
        self.rt_view_group.setExclusive(True)

        self.rt_viewport_pill = QPushButton("3D Viewport")
        self.rt_viewport_pill.setObjectName("subtabPill")
        self.rt_viewport_pill.setCheckable(True)
        self.rt_viewport_pill.setChecked(True)

        self.rt_comparison_pill = QPushButton("Signal Comparison")
        self.rt_comparison_pill.setObjectName("subtabPill")
        self.rt_comparison_pill.setCheckable(True)

        self.rt_view_group.addButton(self.rt_viewport_pill, 0)
        self.rt_view_group.addButton(self.rt_comparison_pill, 1)

        rt_pill_layout.addWidget(self.rt_viewport_pill)
        rt_pill_layout.addWidget(self.rt_comparison_pill)
        rt_pill_layout.addStretch()
        rt_container_layout.addLayout(rt_pill_layout)

        # Stacked widget for viewport vs comparison
        self.rt_view_stack = QStackedWidget()

        # Index 0: viewport page (placeholder, filled by _ensure_sionna_widget)
        self.rt_viewport_page = QWidget()
        self.rt_viewport_page_layout = QVBoxLayout(self.rt_viewport_page)
        self.rt_viewport_page_layout.setContentsMargins(0, 0, 0, 0)
        self.rt_view_stack.addWidget(self.rt_viewport_page)

        # Index 1: RT comparison page
        rt_comparison_frame = QFrame()
        rt_comparison_frame.setObjectName("card")
        rt_comp_layout = QVBoxLayout(rt_comparison_frame)
        rt_comp_layout.setContentsMargins(24, 24, 24, 24)
        rt_comp_title = QLabel("Clean vs Augmented Signal")
        rt_comp_title.setProperty("class", "card-title")
        rt_comp_layout.addWidget(rt_comp_title)

        from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget
        self.rt_comparison_plot = ComparisonWidget()
        rt_comp_layout.addWidget(self.rt_comparison_plot)
        self.rt_view_stack.addWidget(rt_comparison_frame)

        rt_container_layout.addWidget(self.rt_view_stack)

        self.rt_view_group.idClicked.connect(self.rt_view_stack.setCurrentIndex)

        self.rt_right_container.setVisible(False)
        layout.addWidget(self.rt_right_container)

        # ── Comparison card (visible for AWGN / Stochastic) ──
        self.comparison_card = QFrame()
        self.comparison_card.setObjectName("card")
        comparison_layout = QVBoxLayout(self.comparison_card)
        comparison_layout.setContentsMargins(24, 24, 24, 24)

        comparison_title = QLabel("Clean vs Augmented Signal")
        comparison_title.setProperty("class", "card-title")
        comparison_layout.addWidget(comparison_title)

        from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget
        self.comparison_plot = ComparisonWidget()
        comparison_layout.addWidget(self.comparison_plot)

        layout.addWidget(self.comparison_card)

        return panel

    def _ensure_sionna_widget(self) -> bool:
        """Lazy-create the SionnaWidget (viewport_only) on first RT tab activation."""
        if self.sionna_widget is not None:
            return True
        try:
            from mixedsignal_gui.sionna_widget import SionnaWidget
            self.sionna_widget = SionnaWidget(viewport_only=True)
            self.rt_viewport_page_layout.addWidget(self.sionna_widget)

            # Connect signals
            self.sionna_widget.paths_computed.connect(self._on_rt_paths_computed)
            self.sionna_widget.taps_computed.connect(self._on_rt_taps_computed)
            self.sionna_widget.error_occurred.connect(self._on_rt_error)
            self.sionna_widget.scene_loaded.connect(
                lambda path: self.rt_status_label.setText(f"Scene: {path.rsplit('/', 1)[-1]}"))

            # Viewport placement → update position spinboxes
            self.sionna_widget.tx_placed.connect(self._on_rt_tx_placed)
            self.sionna_widget.rx_placed.connect(self._on_rt_rx_placed)

            # Push current control values to the widget
            self._rt_push_max_depth(self.rt_depth_spin.value())
            self._rt_push_num_samples(self.rt_samples_spin.value())
            self._rt_push_tx_position()
            self._rt_push_rx_position()
            self._rt_push_antennas()
            self._rt_push_solver()
            self._rt_push_channel_params()

            return True
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependencies",
                f"Ray Tracing requires TensorFlow and Sionna RT.\n\n{e}"
            )
            return False

    def _on_rt_paths_computed(self, channel_params):
        if channel_params.num_paths == 0:
            self.rt_status_label.setText(
                "Paths: 0 found — check TX/RX positions are inside the scene")
        else:
            self.rt_status_label.setText(f"Paths: {channel_params.num_paths} found")
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

    def _on_rt_taps_computed(self, taps):
        self.rt_last_taps = taps
        self.rt_status_label.setText(f"Taps ready: shape {taps.shape}")

    def _on_rt_error(self, msg):
        self.rt_status_label.setText(f"Error: {msg}")
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

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
        if name and name != "No datasets found":
            entry = self.dataset_manager.get_by_name(name)
            if entry:
                self._active_entry = entry
                self.display_original_signal()
                self._update_stochastic_metadata()
                self._update_rt_metadata()

    def _update_stochastic_metadata(self):
        """Auto-populate read-only carrier freq / sample rate from dataset metadata."""
        entry = self._active_entry
        if entry is None:
            self.carrier_freq_display.setText("--")
            self.sample_rate_display.setText("--")
            return

        fs = entry['fs']
        fc = entry.get('fc', None)

        self.sample_rate_display.setText(f"{fs / 1e6:.4g}")
        self.carrier_freq_display.setText(f"{float(fc) / 1e6:.4g}" if fc is not None else "N/A")

        # Default output_num_samples to signal length
        sig_len = entry['samples']
        self.output_num_samples = sig_len
        self.output_samples_spin.setValue(sig_len)

    def _update_rt_metadata(self):
        """Auto-populate read-only RT fields from dataset metadata."""
        entry = self._active_entry
        if entry is None:
            self.rt_freq_display.setText("--")
            self.rt_sample_rate_display.setText("--")
            return

        fs = entry['fs']
        fc = entry.get('fc', None)

        self.rt_sample_rate_display.setText(f"{fs / 1e6:.4g}")
        self.rt_freq_display.setText(f"{float(fc) / 1e6:.4g}" if fc is not None else "N/A")

    def upload_dataset(self):
        # Get default directory from settings
        settings = QSettings("MyCompany", "MixedSignalGUI")
        default_dir = settings.value("dataPath", "")
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = ""
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import External Dataset", default_dir, "NumPy Files (*.npy);;All Files (*)"
        )
        if not filepath:
            return

        try:
            from PySide6.QtWidgets import QInputDialog
            fs_str, ok = QInputDialog.getText(
                self, "Sample Rate",
                "Enter sample rate (Hz) for this dataset:",
                text="8000000"
            )
            if not ok:
                return
            try:
                fs = float(fs_str)
            except ValueError:
                print("Invalid sample rate — import cancelled.")
                return

            metadata = {
                'source': 'imported',
                'fs':     fs,
            }
            entry = self.dataset_manager.import_external(filepath, metadata)
            self.refresh_dataset_list()
            # Select the newly imported dataset
            self.dataset_combo.setCurrentText(entry['name'])
            print(f"[ChannelTab] Imported: {entry['name']}")

        except Exception as e:
            print(f"Failed to import dataset: {e}")

    def refresh_dataset_list(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        entries = self.dataset_manager.scan()

        if entries:
            names = [e['name'] for e in entries]
            self.dataset_combo.addItems(names)
            # Re-select active entry if it still exists
            if self._active_entry and self._active_entry['name'] in names:
                self.dataset_combo.setCurrentText(self._active_entry['name'])
            else:
                self.dataset_combo.setCurrentIndex(0)
        else:
            self.dataset_combo.addItem("No datasets found")

        self.dataset_combo.blockSignals(False)

        # Trigger a load if we have a valid selection
        current = self.dataset_combo.currentText()
        if current and current != "No datasets found":
            self.on_dataset_changed(current)

        print(f"[ChannelTab] Dataset list refreshed: {self.dataset_combo.count()} entries")

    def display_original_signal(self):
        entry = self._active_entry
        if entry is None:
            print("[ChannelTab] No active dataset to display")
            return

        try:
            self.clean_signal = self.dataset_manager.load_signal(entry)
            self.clean_fs = entry['fs']
            self.clean_metadata = entry   # entire entry is the metadata
        except Exception as e:
            print(f"[ChannelTab] Failed to load signal: {e}")

    # ---- Config builders ----
    def _build_stochastic_config(self):
        entry = self._active_entry or {}
        metadata = entry
        fs = entry.get('fs', 8e6)
        return {
            "seed": self.stoch_seed,
            "waveform": {
                "path": None,
                "format": "npy",
                "mat_key": None,
                "sample_rate_hz": fs,
                "normalize_rms": 1.0,
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
                "max_speed_mps": 0.0,
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

    def _build_rt_config(self):
        """Build multiantenna_config.json-schema dict from SionnaWidget state."""
        self._rt_push_channel_params()
        config = self.sionna_widget.get_rt_config()

        entry = self._active_entry
        if entry:
            fc = entry.get('fc', None)
            if fc is not None:
                config["center_frequency"] = float(fc)
            config["sample_rate"] = entry['fs']

        return config

    def _is_baseband_signal(self, signal):
        """Check if the signal is complex baseband (required for TDL/RT)."""
        return np.iscomplexobj(signal)

    def apply_augmentations(self):
        if not hasattr(self, 'clean_signal'):
            print("No dataset selected to augment.")
            return

        signal = self.clean_signal
        fs = self.clean_fs

        # Stochastic and RT augmentations require complex baseband IQ signals.
        # Real passband signals cast to complex are NOT valid baseband — the
        # carrier frequency relationship is lost and the channel convolution
        # produces physically incorrect results.
        if self.active_subtab in (1, 2) and not self._is_baseband_signal(signal):
            QMessageBox.warning(
                self, "Baseband Signal Required",
                "Stochastic and Ray Tracing channel augmentations operate on "
                "complex baseband (IQ) signals.\n\n"
                "The selected dataset contains a real passband signal, which "
                "cannot be correctly processed by these channel models.\n\n"
                "To fix this, go to the Waveform Generation tab and set the "
                "output type to \"Baseband (Complex IQ)\", then regenerate."
            )
            return

        if self.active_subtab == 1:
            # Stochastic TDL path — sample ranges
            try:
                rng = np.random.default_rng()
                sampled_delay = self.delay_spread_spin.value_or_range().sample(rng)
                sampled_stoch_snr = self.stoch_snr_spin.value_or_range().sample(rng)

                multi_ch = self.stoch_multi_channel_cb.isChecked()
                config = self._build_stochastic_config()
                # Override with sampled values
                config["channel"]["delay_spread_s"] = sampled_delay * 1e-9
                config["noise"]["snr_db"] = sampled_stoch_snr
                if multi_ch:
                    config["channel"]["num_tx_ant"] = self.stoch_num_tx_ant_spin.value()
                    config["channel"]["num_rx_ant"] = self.stoch_num_rx_ant_spin.value()
                block = StochasticTDLAugmentation(config, multi_channel=multi_ch)
                augmented_signal = block.apply(signal, fs)

                self.last_augmentation_config = {
                    'delay_spread_ns': self.delay_spread_spin.value_or_range().to_metadata(sampled_delay),
                    'stoch_snr_db': self.stoch_snr_spin.value_or_range().to_metadata(sampled_stoch_snr),
                    'stoch_config': config,
                }
                if multi_ch and augmented_signal.ndim == 2:
                    self.stoch_output_shape_label.setText(
                        f"Output: {augmented_signal.shape[0]} antennas x {augmented_signal.shape[1]} samples")
                else:
                    self.stoch_output_shape_label.setText("")
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
        elif self.active_subtab == 2:
            # Ray Tracing path
            if self.rt_last_taps is None:
                QMessageBox.warning(
                    self, "No Taps",
                    "Load a scene and compute paths first."
                )
                return
            try:
                multi_ch = self.rt_multi_channel_cb.isChecked()
                config = self._build_rt_config()
                block = SionnaRTAugmentation(config, self.rt_last_taps, multi_channel=multi_ch)
                augmented_signal = block.apply(signal, fs)
                if multi_ch and augmented_signal.ndim == 2:
                    self.rt_output_shape_label.setText(
                        f"Output: {augmented_signal.shape[0]} antennas x {augmented_signal.shape[1]} samples")
                else:
                    self.rt_output_shape_label.setText("")
            except ImportError as e:
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    f"RT augmentation requires TensorFlow and Sionna.\n\n{e}"
                )
                return
            except Exception as e:
                QMessageBox.critical(
                    self, "Augmentation Error",
                    f"RT augmentation failed:\n\n{e}"
                )
                return
        else:
            # AWGN path — sample ranges once per apply
            rng = np.random.default_rng()
            sampled_snr = self.snr_spin.value_or_range().sample(rng)
            sampled_amp = self.amplitude_spin.value_or_range().sample(rng)
            sampled_phase = self.phase_spin.value_or_range().sample(rng)
            sampled_freq_hz = self.freq_shift_spin.value_or_range().sample(rng)

            pipeline = AugmentationPipeline()
            if self.awgn_enabled:
                pipeline.add(AWGNAugmentation(snr_db=sampled_snr))
            if self.amp_phase_enabled:
                phase_rad = np.deg2rad(sampled_phase)
                pipeline.add(ScalarAmplitudeAndPhaseShift(amplitude=sampled_amp, phi=phase_rad))
            if self.freq_shift_enabled:
                pipeline.add(FrequencyShift(delta_f=sampled_freq_hz))
            augmented_signal = pipeline.apply(signal, fs)

            # Record range + sampled values for save_augmented_dataset
            self.last_augmentation_config = {
                'awgn': {'enabled': self.awgn_enabled,
                         'snr_db': self.snr_spin.value_or_range().to_metadata(sampled_snr)},
                'amp_phase': {'enabled': self.amp_phase_enabled,
                              'amplitude': self.amplitude_spin.value_or_range().to_metadata(sampled_amp),
                              'phase_deg': self.phase_spin.value_or_range().to_metadata(sampled_phase)},
                'freq_shift': {'enabled': self.freq_shift_enabled,
                               'freq_shift_hz': self.freq_shift_spin.value_or_range().to_metadata(sampled_freq_hz)},
            }

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

        # Choose the correct comparison widget
        if self.active_subtab == 2:
            plot_widget = self.rt_comparison_plot
            # Auto-switch to comparison view
            self.rt_view_stack.setCurrentIndex(1)
            self.rt_comparison_pill.setChecked(True)
        else:
            plot_widget = self.comparison_plot

        # Display comparison
        plot_widget.plot_comparison(
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
            print("[ChannelTab] No augmented signal to save. Apply augmentations first.")
            return

        entry = self._active_entry
        base_name = entry['name'] if entry else "dataset"
        aug_name = self.dataset_manager._unique_name(f"{base_name}_augmented")

        # Build augmentation metadata on top of original entry
        metadata = dict(entry) if entry else {}
        # Drop internal path keys — they will be re-set by save()
        metadata.pop('_npy_path', None)
        metadata.pop('_json_path', None)

        metadata['augmented'] = True
        metadata['base_dataset'] = base_name
        metadata['fs'] = self.last_augmented_fs
        metadata['source'] = 'augmented'

        if self.active_subtab == 1:
            metadata['augmentation_type'] = 'stochastic_tdl'
            if hasattr(self, 'last_augmentation_config'):
                metadata['augmentation_config'] = self.last_augmentation_config
            else:
                aug_config = self._build_stochastic_config()
                aug_config["waveform"]["path"] = f"{aug_name}.npy"
                metadata['augmentation_config'] = aug_config
        elif self.active_subtab == 2:
            aug_config = self._build_rt_config()
            aug_config["transmitters"][0]["waveform_path"] = f"{aug_name}.npy"
            metadata['augmentation_type'] = 'sionna_rt'
            metadata['augmentation_config'] = aug_config
        else:
            metadata['augmentation_type'] = 'awgn'
            if hasattr(self, 'last_augmentation_config'):
                metadata['augmentation_config'] = self.last_augmentation_config
            else:
                metadata['augmentation_config'] = {
                    'awgn':      {'enabled': self.awgn_enabled, 'snr_db': self.snr_db},
                    'amp_phase': {'enabled': self.amp_phase_enabled,
                                  'amplitude': self.amplitude, 'phase_deg': self.phase_deg},
                    'freq_shift': {'enabled': self.freq_shift_enabled,
                                   'freq_shift_hz': self.freq_shift_hz},
                }

        # Determine save mode for multi-channel signals
        signal = self.last_augmented_signal
        save_mode = "Single Antenna"
        if self.active_subtab == 1 and self.stoch_multi_channel_cb.isChecked():
            save_mode = self.stoch_save_mode_combo.currentText()
        elif self.active_subtab == 2 and self.rt_multi_channel_cb.isChecked():
            save_mode = self.rt_save_mode_combo.currentText()

        if save_mode == "All Antennas (Separate Files)" and signal.ndim == 2:
            num_ant = signal.shape[0]
            metadata['num_channels'] = num_ant
            for i in range(num_ant):
                ant_name = self.dataset_manager._unique_name(f"{base_name}_augmented_ant{i}")
                meta_i = dict(metadata)
                meta_i['antenna_index'] = i
                meta_i['channel_group'] = aug_name
                saved = self.dataset_manager.save(ant_name, signal[i], meta_i)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved {num_ant} per-antenna datasets: {aug_name}_ant*")
        elif save_mode == "All Antennas (Stacked)" and signal.ndim == 2:
            metadata['num_channels'] = signal.shape[0]
            metadata['signal_shape'] = list(signal.shape)
            saved = self.dataset_manager.save(aug_name, signal, metadata)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved stacked multi-channel dataset: {aug_name} {signal.shape}")
        else:
            # Single antenna or 1D signal — original behavior
            saved = self.dataset_manager.save(aug_name, signal, metadata)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved augmented dataset: {aug_name}")

    # ------------------------------------------------------------------
    # Bulk augmentation
    # ------------------------------------------------------------------

    def apply_to_all_in_folder(self):
        """Run the current augmentation config over a user-chosen folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Datasets Folder",
            str(self.dataset_manager.datasets_dir))
        if not folder:
            return

        source = DatasetManager(datasets_dir=folder)
        entries = source.scan()
        if not entries:
            QMessageBox.information(self, "Nothing to Process",
                                    "The selected folder contains no datasets.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = os.path.join(folder, f"augmented_{ts}")

        reply = QMessageBox.question(
            self, "Bulk Augmentation",
            f"Apply augmentations to all {len(entries)} datasets?\n\n"
            f"Output folder:\n{dest_dir}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        dest = DatasetManager(datasets_dir=dest_dir)

        # Build ranges + static config from current widget state
        ranges: dict[str, ParameterRange] = {}
        static_config: dict = {}
        rt_taps = None

        if self.active_subtab == 0:
            kind = "awgn"
            ranges["snr_db"] = self.snr_spin.value_or_range()
            ranges["amplitude"] = self.amplitude_spin.value_or_range()
            ranges["phase_deg"] = self.phase_spin.value_or_range()
            ranges["freq_shift_hz"] = self.freq_shift_spin.value_or_range()
            static_config["awgn_enabled"] = self.awgn_enabled
            static_config["amp_phase_enabled"] = self.amp_phase_enabled
            static_config["freq_shift_enabled"] = self.freq_shift_enabled

        elif self.active_subtab == 1:
            kind = "stoch_tdl"
            ranges["delay_spread_ns"] = self.delay_spread_spin.value_or_range()
            ranges["stoch_snr_db"] = self.stoch_snr_spin.value_or_range()
            static_config["stoch_config"] = self._build_stochastic_config()
            multi_ch = self.stoch_multi_channel_cb.isChecked()
            static_config["multi_channel"] = multi_ch
            if multi_ch:
                static_config["stoch_config"]["channel"]["num_tx_ant"] = \
                    self.stoch_num_tx_ant_spin.value()
                static_config["stoch_config"]["channel"]["num_rx_ant"] = \
                    self.stoch_num_rx_ant_spin.value()

        elif self.active_subtab == 2:
            kind = "rt"
            if self.rt_last_taps is None:
                QMessageBox.warning(self, "No Taps",
                                    "Load a scene and compute paths first.")
                return
            rt_taps = self.rt_last_taps
            static_config["rt_config"] = self._build_rt_config()
            multi_ch = self.rt_multi_channel_cb.isChecked()
            static_config["multi_channel"] = multi_ch
        else:
            return

        self._bulk_thread = BulkAugmentationThread(
            source_manager=source,
            dest_manager=dest,
            augmentation_kind=kind,
            ranges=ranges,
            static_config=static_config,
            rt_taps=rt_taps,
        )

        # Progress dialog
        self._bulk_progress = QProgressDialog(
            "Augmenting datasets…", "Cancel", 0, len(entries), self)
        self._bulk_progress.setWindowTitle("Bulk Augmentation")
        self._bulk_progress.setMinimumDuration(0)
        self._bulk_progress.setModal(True)

        self._bulk_progress.canceled.connect(self._bulk_thread.request_cancel)
        self._bulk_thread.progress.connect(self._on_bulk_progress)
        self._bulk_thread.finished.connect(self._on_bulk_finished)
        self._bulk_thread.error.connect(self._on_bulk_error)

        self._bulk_thread.start()

    def _on_bulk_progress(self, current: int, total: int, name: str):
        self._bulk_progress.setMaximum(total)
        self._bulk_progress.setValue(current)
        self._bulk_progress.setLabelText(f"Processing {current}/{total}: {name}")

    def _on_bulk_finished(self, ok: int, errors: int, dest_dir: str):
        self._bulk_progress.close()
        QMessageBox.information(
            self, "Bulk Augmentation Complete",
            f"Processed: {ok + errors}\n"
            f"Succeeded: {ok}\n"
            f"Errors: {errors}\n\n"
            f"Output: {dest_dir}",
        )
        self.refresh_dataset_list()

    def _on_bulk_error(self, msg: str):
        self._bulk_progress.close()
        QMessageBox.critical(self, "Bulk Augmentation Error", msg)