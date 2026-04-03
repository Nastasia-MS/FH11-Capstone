"""Lightweight Sionna RT channel widget for PySide6 applications.

Usage::

    from sionna_widget import SionnaWidget, SionnaChannelAugmentation, ChannelParameters

    widget = SionnaWidget()
    layout.addWidget(widget)
"""

from .widget import SionnaWidget
from .augmentation import SionnaChannelAugmentation, ChannelParameters

__all__ = ["SionnaWidget", "SionnaChannelAugmentation", "ChannelParameters"]
