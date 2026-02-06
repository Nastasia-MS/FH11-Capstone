"""
Tab modules for the Signal Generation & Classification dashboard
"""

from .waveform_tab import WaveformSelectionTab
from .channel_tab import ChannelNoiseTab
from .ml_training_tab import MLTrainingTab
from .inference_tab import InferenceResultsTab

__all__ = [
    'WaveformSelectionTab',
    'ChannelNoiseTab',
    'MLTrainingTab',
    'InferenceResultsTab'
]