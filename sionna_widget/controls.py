"""Control panel for the Sionna widget.

Provides scene loading, simulation parameters, TX/RX positioning,
antenna array configuration, PathSolver options, channel/taps parameters,
and a compute button.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QLabel, QScrollArea, QHBoxLayout, QCheckBox, QComboBox,
)
from PySide6.QtCore import Signal


class Vector3Input(QWidget):
    """Reusable XYZ input widget."""
    value_changed = Signal(list)

    def __init__(self, default=(0, 0, 0), decimals=3, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        for i, label in enumerate(("X", "Y", "Z")):
            spin = QDoubleSpinBox()
            spin.setPrefix(f"{label}: ")
            spin.setRange(-10000, 10000)
            spin.setDecimals(decimals)
            spin.setValue(default[i])
            spin.valueChanged.connect(self._emit_change)
            layout.addWidget(spin)
            self.spins.append(spin)

    def get_value(self) -> list:
        return [s.value() for s in self.spins]

    def set_value(self, vals):
        for s, v in zip(self.spins, vals):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)

    def _emit_change(self):
        self.value_changed.emit(self.get_value())


class SimpleControlPanel(QWidget):
    """Compact control panel: scene, parameters, TX/RX positions, antenna arrays,
    solver options, channel parameters, compute."""

    config_changed = Signal()
    tx_position_changed = Signal(list)
    rx_position_changed = Signal(list)
    load_scene_requested = Signal()
    compute_requested = Signal()
    place_tx_requested = Signal()
    place_rx_requested = Signal()
    antenna_config_changed = Signal()
    solver_options_changed = Signal()
    channel_params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        main_layout = QVBoxLayout(container)

        # -- Scene group ---------------------------------------------------
        scene_group = QGroupBox("Scene")
        scene_vbox = QVBoxLayout(scene_group)
        self._scene_label = QLabel("No scene loaded")
        self._scene_label.setWordWrap(True)
        scene_vbox.addWidget(self._scene_label)

        load_btn = QPushButton("Load Scene...")
        load_btn.clicked.connect(self.load_scene_requested)
        scene_vbox.addWidget(load_btn)
        main_layout.addWidget(scene_group)

        # -- Parameters group ----------------------------------------------
        param_group = QGroupBox("Parameters")
        param_form = QFormLayout(param_group)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 100.0)
        self.freq_spin.setValue(3.5)
        self.freq_spin.setSuffix(" GHz")
        self.freq_spin.setDecimals(2)
        self.freq_spin.valueChanged.connect(self.config_changed)
        param_form.addRow("Frequency:", self.freq_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(5)
        self.depth_spin.valueChanged.connect(self.config_changed)
        param_form.addRow("Max Depth:", self.depth_spin)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1000, 10_000_000)
        self.samples_spin.setSingleStep(100_000)
        self.samples_spin.setValue(1_000_000)
        self.samples_spin.valueChanged.connect(self.config_changed)
        param_form.addRow("Samples:", self.samples_spin)

        main_layout.addWidget(param_group)

        # -- Transceivers group --------------------------------------------
        trx_group = QGroupBox("Transceivers")
        trx_form = QFormLayout(trx_group)

        trx_form.addRow(QLabel("TX Position:"))
        self.tx_pos_input = Vector3Input(default=(0, 0, 10), decimals=1)
        self.tx_pos_input.value_changed.connect(self.tx_position_changed)
        trx_form.addRow(self.tx_pos_input)

        self._place_tx_btn = QPushButton("Click to Place TX")
        self._place_tx_btn.setCheckable(True)
        self._place_tx_btn.toggled.connect(self._on_place_tx_toggled)
        trx_form.addRow(self._place_tx_btn)

        trx_form.addRow(QLabel("RX Position:"))
        self.rx_pos_input = Vector3Input(default=(50, 0, 1.5), decimals=1)
        self.rx_pos_input.value_changed.connect(self.rx_position_changed)
        trx_form.addRow(self.rx_pos_input)

        self._place_rx_btn = QPushButton("Click to Place RX")
        self._place_rx_btn.setCheckable(True)
        self._place_rx_btn.toggled.connect(self._on_place_rx_toggled)
        trx_form.addRow(self._place_rx_btn)

        main_layout.addWidget(trx_group)

        # -- Antenna Arrays group ------------------------------------------
        self._build_antenna_group(main_layout)

        # -- Path Solver group ---------------------------------------------
        self._build_solver_group(main_layout)

        # -- Channel Parameters group --------------------------------------
        self._build_channel_group(main_layout)

        # -- Compute button ------------------------------------------------
        self.compute_btn = QPushButton("Compute Paths")
        self.compute_btn.setStyleSheet(
            "QPushButton { background-color: #2d8c3c; color: white; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3aa64e; }"
        )
        self.compute_btn.clicked.connect(self.compute_requested)
        main_layout.addWidget(self.compute_btn)

        main_layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- Antenna Arrays ----------------------------------------------------

    def _build_antenna_group(self, parent_layout):
        group = QGroupBox("Antenna Arrays")
        form = QFormLayout(group)

        # TX antenna
        form.addRow(QLabel("TX Array"))
        self.tx_ant_rows = QSpinBox()
        self.tx_ant_rows.setRange(1, 16)
        self.tx_ant_rows.setValue(1)
        self.tx_ant_rows.valueChanged.connect(self._on_antenna_changed)
        form.addRow("  Rows:", self.tx_ant_rows)

        self.tx_ant_cols = QSpinBox()
        self.tx_ant_cols.setRange(1, 16)
        self.tx_ant_cols.setValue(1)
        self.tx_ant_cols.valueChanged.connect(self._on_antenna_changed)
        form.addRow("  Cols:", self.tx_ant_cols)

        self.tx_ant_pattern = QComboBox()
        self.tx_ant_pattern.addItems(["iso", "tr38901"])
        self.tx_ant_pattern.currentTextChanged.connect(
            lambda _: self._on_antenna_changed())
        form.addRow("  Pattern:", self.tx_ant_pattern)

        self.tx_ant_pol = QComboBox()
        self.tx_ant_pol.addItems(["V", "H", "cross"])
        self.tx_ant_pol.currentTextChanged.connect(
            lambda _: self._on_antenna_changed())
        form.addRow("  Polarization:", self.tx_ant_pol)

        # RX antenna
        form.addRow(QLabel("RX Array"))
        self.rx_ant_rows = QSpinBox()
        self.rx_ant_rows.setRange(1, 16)
        self.rx_ant_rows.setValue(1)
        self.rx_ant_rows.valueChanged.connect(self._on_antenna_changed)
        form.addRow("  Rows:", self.rx_ant_rows)

        self.rx_ant_cols = QSpinBox()
        self.rx_ant_cols.setRange(1, 16)
        self.rx_ant_cols.setValue(1)
        self.rx_ant_cols.valueChanged.connect(self._on_antenna_changed)
        form.addRow("  Cols:", self.rx_ant_cols)

        self.rx_ant_pattern = QComboBox()
        self.rx_ant_pattern.addItems(["iso", "tr38901"])
        self.rx_ant_pattern.currentTextChanged.connect(
            lambda _: self._on_antenna_changed())
        form.addRow("  Pattern:", self.rx_ant_pattern)

        self.rx_ant_pol = QComboBox()
        self.rx_ant_pol.addItems(["V", "H", "cross"])
        self.rx_ant_pol.currentTextChanged.connect(
            lambda _: self._on_antenna_changed())
        form.addRow("  Polarization:", self.rx_ant_pol)

        parent_layout.addWidget(group)

    def _on_antenna_changed(self, _=None):
        self._update_rx_antenna_index_max()
        self.antenna_config_changed.emit()

    # -- Path Solver -------------------------------------------------------

    def _build_solver_group(self, parent_layout):
        group = QGroupBox("Path Solver")
        vbox = QVBoxLayout(group)

        self.cb_los = QCheckBox("Line of Sight")
        self.cb_los.setChecked(True)
        self.cb_los.toggled.connect(lambda _: self.solver_options_changed.emit())
        vbox.addWidget(self.cb_los)

        self.cb_specular = QCheckBox("Specular Reflection")
        self.cb_specular.setChecked(True)
        self.cb_specular.toggled.connect(lambda _: self.solver_options_changed.emit())
        vbox.addWidget(self.cb_specular)

        self.cb_diffuse = QCheckBox("Diffuse Reflection")
        self.cb_diffuse.setChecked(True)
        self.cb_diffuse.toggled.connect(lambda _: self.solver_options_changed.emit())
        vbox.addWidget(self.cb_diffuse)

        self.cb_refraction = QCheckBox("Refraction")
        self.cb_refraction.setChecked(True)
        self.cb_refraction.toggled.connect(lambda _: self.solver_options_changed.emit())
        vbox.addWidget(self.cb_refraction)

        self.cb_synthetic = QCheckBox("Synthetic Array")
        self.cb_synthetic.setChecked(True)
        self.cb_synthetic.toggled.connect(lambda _: self.solver_options_changed.emit())
        vbox.addWidget(self.cb_synthetic)

        parent_layout.addWidget(group)

    # -- Channel Parameters ------------------------------------------------

    def _build_channel_group(self, parent_layout):
        group = QGroupBox("Channel Parameters")
        form = QFormLayout(group)

        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.01, 1000.0)
        self.bandwidth_spin.setValue(5.0)
        self.bandwidth_spin.setSuffix(" MHz")
        self.bandwidth_spin.setDecimals(2)
        self.bandwidth_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("Bandwidth:", self.bandwidth_spin)

        self.l_min_spin = QSpinBox()
        self.l_min_spin.setRange(0, 10000)
        self.l_min_spin.setValue(0)
        self.l_min_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("l_min:", self.l_min_spin)

        self.l_max_spin = QSpinBox()
        self.l_max_spin.setRange(1, 10000)
        self.l_max_spin.setValue(200)
        self.l_max_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("l_max:", self.l_max_spin)

        self.tx_power_spin = QDoubleSpinBox()
        self.tx_power_spin.setRange(-30.0, 60.0)
        self.tx_power_spin.setValue(30.0)
        self.tx_power_spin.setSuffix(" dBm")
        self.tx_power_spin.setDecimals(1)
        self.tx_power_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("TX Power:", self.tx_power_spin)

        self.noise_power_spin = QDoubleSpinBox()
        self.noise_power_spin.setRange(-200.0, 0.0)
        self.noise_power_spin.setValue(-108.0)
        self.noise_power_spin.setSuffix(" dBm")
        self.noise_power_spin.setDecimals(1)
        self.noise_power_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("Noise Power:", self.noise_power_spin)

        self.waveform_length_spin = QSpinBox()
        self.waveform_length_spin.setRange(64, 65536)
        self.waveform_length_spin.setValue(1024)
        self.waveform_length_spin.setSingleStep(256)
        self.waveform_length_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("Waveform Length:", self.waveform_length_spin)

        self.rx_antenna_idx_spin = QSpinBox()
        self.rx_antenna_idx_spin.setRange(0, 0)
        self.rx_antenna_idx_spin.setValue(0)
        self.rx_antenna_idx_spin.valueChanged.connect(
            lambda _: self.channel_params_changed.emit())
        form.addRow("RX Antenna Idx:", self.rx_antenna_idx_spin)

        parent_layout.addWidget(group)

    def _update_rx_antenna_index_max(self):
        max_idx = self.rx_ant_rows.value() * self.rx_ant_cols.value() - 1
        self.rx_antenna_idx_spin.setMaximum(max(max_idx, 0))
        if self.rx_antenna_idx_spin.value() > max_idx:
            self.rx_antenna_idx_spin.setValue(0)

    # -- helpers -----------------------------------------------------------

    def set_scene_label(self, text: str):
        self._scene_label.setText(text)

    def set_tx_position(self, pos: list):
        self.tx_pos_input.set_value(pos)

    def set_rx_position(self, pos: list):
        self.rx_pos_input.set_value(pos)

    def uncheck_placement_buttons(self):
        """Reset both placement toggles (called when placement finishes)."""
        self._place_tx_btn.blockSignals(True)
        self._place_rx_btn.blockSignals(True)
        self._place_tx_btn.setChecked(False)
        self._place_rx_btn.setChecked(False)
        self._place_tx_btn.blockSignals(False)
        self._place_rx_btn.blockSignals(False)

    def get_frequency_hz(self) -> float:
        return self.freq_spin.value() * 1e9

    def get_max_depth(self) -> int:
        return self.depth_spin.value()

    def get_num_samples(self) -> int:
        return self.samples_spin.value()

    # -- antenna getters ---------------------------------------------------

    def get_tx_antenna_config(self) -> dict:
        return {
            "num_rows": self.tx_ant_rows.value(),
            "num_cols": self.tx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.tx_ant_pattern.currentText(),
            "polarization": self.tx_ant_pol.currentText(),
        }

    def get_rx_antenna_config(self) -> dict:
        return {
            "num_rows": self.rx_ant_rows.value(),
            "num_cols": self.rx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.rx_ant_pattern.currentText(),
            "polarization": self.rx_ant_pol.currentText(),
        }

    # -- solver getters ----------------------------------------------------

    def get_solver_options(self) -> dict:
        return {
            "line_of_sight": self.cb_los.isChecked(),
            "specular_reflection": self.cb_specular.isChecked(),
            "diffuse_reflection": self.cb_diffuse.isChecked(),
            "refraction": self.cb_refraction.isChecked(),
            "synthetic_array": self.cb_synthetic.isChecked(),
        }

    # -- channel param getters ---------------------------------------------

    def get_channel_params(self) -> dict:
        return {
            "bandwidth_hz": self.bandwidth_spin.value() * 1e6,
            "l_min": self.l_min_spin.value(),
            "l_max": self.l_max_spin.value(),
            "tx_power_dbm": self.tx_power_spin.value(),
            "noise_power_dbm": self.noise_power_spin.value(),
            "waveform_length": self.waveform_length_spin.value(),
            "rx_antenna_index": self.rx_antenna_idx_spin.value(),
        }

    # -- toggle logic ------------------------------------------------------

    def _on_place_tx_toggled(self, on: bool):
        if on:
            self._place_rx_btn.blockSignals(True)
            self._place_rx_btn.setChecked(False)
            self._place_rx_btn.blockSignals(False)
            self.place_tx_requested.emit()
        # Exiting is handled by the widget when placement completes

    def _on_place_rx_toggled(self, on: bool):
        if on:
            self._place_tx_btn.blockSignals(True)
            self._place_tx_btn.setChecked(False)
            self._place_tx_btn.blockSignals(False)
            self.place_rx_requested.emit()
