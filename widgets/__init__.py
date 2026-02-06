"""
Custom widget modules for the Signal Generation & Classification dashboard
"""

from .constellation import ConstellationWidget
from .power_spectrum import PowerSpectrumWidget
from .noise_spectrum import NoiseSpectrumWidget
from .toggle_switch import ToggleSwitch
from .training_chart import TrainingChartWidget

__all__ = [
    'ConstellationWidget',
    'PowerSpectrumWidget',
    'NoiseSpectrumWidget',
    'ToggleSwitch',
    'TrainingChartWidget'
]