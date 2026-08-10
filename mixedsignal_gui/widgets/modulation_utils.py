"""Shared helpers for modulation-selection widgets.

Used by the Waveform and Evaluate Model tabs, which offer the same list of
waveform types and must agree on which of them the current engine can
actually produce.
"""

from PySide6.QtCore import Qt

from mixedsignal_gui.backend.generators import unavailable_modulations


def selected_modulation(combo) -> str:
    """The modulation name currently chosen, without any UI annotation.

    ``mark_unavailable_modulations`` rewrites entries to "WiFi  (needs MATLAB)",
    so reading ``currentText()`` directly would yield a name no generator
    recognises.  Disabled entries cannot normally be selected, but stripping
    here means the annotation can never leak into a config.
    """
    return combo.currentText().split()[0]


def mark_unavailable_modulations(combo, matlab_engine):
    """Grey out modulations the current engine cannot produce, and say why.

    Without this the only feedback is an error dialog after pressing Generate.
    Disabling the entry up front means an unavailable waveform cannot be picked
    by accident, and the tooltip names the missing toolbox.

    A no-op when MATLAB is running, since everything is available then.
    """
    blocked = unavailable_modulations(matlab_engine)
    if not blocked:
        return

    model = combo.model()
    for index in range(combo.count()):
        # Match on the leading token so re-running this is harmless.
        name = combo.itemText(index).split()[0]
        reason = blocked.get(name)
        if reason is None:
            continue

        combo.setItemText(index, f"{name}  (needs MATLAB)")
        combo.setItemData(index, f"Requires MATLAB and the {reason}.", Qt.ToolTipRole)

        # QComboBox uses a QStandardItemModel by default, which supports
        # per-item enabling; guard in case that ever changes.
        item = model.item(index) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(False)
