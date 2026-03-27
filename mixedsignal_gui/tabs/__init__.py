"""
Tab modules for the Signal Generation & Classification dashboard
"""

from mixedsignal_gui.tabs.waveform_tab import WaveformSelectionTab
from mixedsignal_gui.tabs.channel_tab import ChannelNoiseTab
from mixedsignal_gui.tabs.ml_training_tab import MLTrainingTab
from mixedsignal_gui.tabs.inference_tab import InferenceResultsTab

__all__ = [
    'WaveformSelectionTab',
    'ChannelNoiseTab',
    'MLTrainingTab',
    'InferenceResultsTab'
]