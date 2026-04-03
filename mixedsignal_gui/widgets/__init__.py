"""
Custom widget modules for the Signal Generation & Classification dashboard
"""

from mixedsignal_gui.widgets.constellation import ConstellationWidget
from mixedsignal_gui.widgets.power_spectrum import PowerSpectrumWidget
from mixedsignal_gui.widgets.noise_spectrum import NoiseSpectrumWidget
from mixedsignal_gui.widgets.toggle_switch import ToggleSwitch
from mixedsignal_gui.widgets.training_chart import TrainingChartWidget

__all__ = [
    'ConstellationWidget',
    'PowerSpectrumWidget',
    'NoiseSpectrumWidget',
    'ToggleSwitch',
    'TrainingChartWidget'
]