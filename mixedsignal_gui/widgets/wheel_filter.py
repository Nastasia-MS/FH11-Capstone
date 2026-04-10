from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox


class WheelBlocker(QObject):
    """Block mouse wheel changes on spin boxes and combo boxes."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            return True
        return super().eventFilter(obj, event)


def install_wheel_blocker(root_widget):
    """Install a wheel blocker on all existing editable controls under a widget."""
    blocker = WheelBlocker(root_widget)
    root_widget._wheel_blocker = blocker

    for widget in root_widget.findChildren(QAbstractSpinBox):
        widget.installEventFilter(blocker)
    for widget in root_widget.findChildren(QComboBox):
        widget.installEventFilter(blocker)
