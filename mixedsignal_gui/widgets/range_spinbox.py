"""
RangeSpinBox — drop-in replacement for QDoubleSpinBox that optionally
lets the user supply a [min, max] uniform range instead of a single value.

The "Vary" checkbox is created internally but NOT placed in the widget's
own layout.  Callers retrieve it via the `vary_checkbox` property and
place it wherever makes sense (e.g. next to the parameter label).
"""

from __future__ import annotations

from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QStackedWidget,
    QWidget,
)

from mixedsignal_gui.backend.parameter_range import ParameterRange


class RangeSpinBox(QWidget):
    """Single-value *or* uniform-range spinbox with a "Vary" toggle."""

    valueChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # --- "Vary" checkbox (NOT in this layout — caller places it) ---
        self._vary_cb = QCheckBox("Range")
        self._vary_cb.setToolTip("Toggle uniform-random range mode")

        # --- Stacked widget: page 0 = fixed, page 1 = range ---
        self._stack = QStackedWidget()

        # Page 0 — single value, right-aligned via stretch
        fixed_page = QWidget()
        fixed_lay = QHBoxLayout(fixed_page)
        fixed_lay.setContentsMargins(0, 0, 0, 0)
        self._fixed_spin = QDoubleSpinBox()
        self._fixed_spin.setFixedWidth(150)
        fixed_lay.addStretch()
        fixed_lay.addWidget(self._fixed_spin)
        self._stack.addWidget(fixed_page)

        # Page 1 — min / max pair, right-aligned
        range_page = QWidget()
        range_lay = QHBoxLayout(range_page)
        range_lay.setContentsMargins(0, 0, 0, 0)
        range_lay.setSpacing(6)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setPrefix("Min: ")
        self._min_spin.setFixedWidth(150)
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setPrefix("Max: ")
        self._max_spin.setFixedWidth(150)
        range_lay.addStretch()
        range_lay.addWidget(self._min_spin)
        range_lay.addWidget(self._max_spin)
        self._stack.addWidget(range_page)

        root.addWidget(self._stack)

        # --- Wiring ---
        self._vary_cb.toggled.connect(self._on_vary_toggled)
        self._fixed_spin.valueChanged.connect(self._emit_changed)
        self._min_spin.valueChanged.connect(self._emit_changed)
        self._min_spin.valueChanged.connect(self._clamp_min)
        self._max_spin.valueChanged.connect(self._emit_changed)
        self._max_spin.valueChanged.connect(self._clamp_max)

    # ------------------------------------------------------------------
    # Public: checkbox for external placement
    # ------------------------------------------------------------------
    @property
    def vary_checkbox(self) -> QCheckBox:
        """The Vary checkbox — caller should addWidget() it next to the label."""
        return self._vary_cb

    # ------------------------------------------------------------------
    # QDoubleSpinBox-compatible forwarding API
    # ------------------------------------------------------------------
    def setRange(self, min_: float, max_: float) -> None:
        for spin in (self._fixed_spin, self._min_spin, self._max_spin):
            spin.setRange(min_, max_)

    def setValue(self, v: float) -> None:
        self._fixed_spin.setValue(v)
        self._min_spin.setValue(v)
        self._max_spin.setValue(v)

    def setSingleStep(self, s: float) -> None:
        for spin in (self._fixed_spin, self._min_spin, self._max_spin):
            spin.setSingleStep(s)

    def setDecimals(self, d: int) -> None:
        for spin in (self._fixed_spin, self._min_spin, self._max_spin):
            spin.setDecimals(d)

    def setSuffix(self, s: str) -> None:
        for spin in (self._fixed_spin, self._min_spin, self._max_spin):
            spin.setSuffix(s)

    def value(self) -> float:
        return self._fixed_spin.value()

    # ------------------------------------------------------------------
    # Range-aware API
    # ------------------------------------------------------------------
    def value_or_range(self) -> ParameterRange:
        if self._vary_cb.isChecked():
            return ParameterRange.uniform(self._min_spin.value(),
                                          self._max_spin.value())
        return ParameterRange.fixed(self._fixed_spin.value())

    def isFixed(self) -> bool:
        return not self._vary_cb.isChecked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_vary_toggled(self, checked: bool) -> None:
        self._stack.setCurrentIndex(1 if checked else 0)
        self._emit_changed()

    def _clamp_min(self, v: float) -> None:
        if v > self._max_spin.value():
            with QSignalBlocker(self._max_spin):
                self._max_spin.setValue(v)

    def _clamp_max(self, v: float) -> None:
        if v < self._min_spin.value():
            with QSignalBlocker(self._min_spin):
                self._min_spin.setValue(v)

    def _emit_changed(self, _value: float | None = None) -> None:
        if self._vary_cb.isChecked():
            self.valueChanged.emit(self._min_spin.value())
        else:
            self.valueChanged.emit(self._fixed_spin.value())
