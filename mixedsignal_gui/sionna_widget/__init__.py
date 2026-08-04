"""Lightweight Sionna RT channel widget for PySide6 applications.

Usage::

    from sionna_widget import SionnaWidget, SionnaChannelAugmentation, ChannelParameters

    widget = SionnaWidget()
    layout.addWidget(widget)

Importing these names pulls in Sionna and TensorFlow, so they are resolved
lazily (PEP 562).  That lets light-weight submodules such as ``scenes`` be
imported without paying the cost — ``channel_tab`` reads the bundled-scene
registry at start-up but only constructs a ``SionnaWidget`` when the user
opens the Ray Tracing sub-tab.
"""

__all__ = ["SionnaWidget", "SionnaChannelAugmentation", "ChannelParameters"]

_LAZY = {
    "SionnaWidget": ".widget",
    "SionnaChannelAugmentation": ".augmentation",
    "ChannelParameters": ".augmentation",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(list(globals()) + __all__)
