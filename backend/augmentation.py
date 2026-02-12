"""
RF Signal Augmentation Pipeline for Machine Learning
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from scipy.signal import hilbert
import numpy as np


class AugmentationBlock(ABC):
    """Base class for all augmentation blocks"""

    @abstractmethod
    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """
        Apply augmentation to a signal.

        Args:
            signal: Input signal (real passband from MATLAB generator)
            fs: Sample rate in Hz
            **kwargs: Additional parameters

        Returns:
            Augmented signal
        """
        pass

    def __call__(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """Allow calling block as a function"""
        return self.apply(signal, fs, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AugmentationPipeline:
    """class to chain multiple augmentation blocks together"""

    def __init__(self, blocks: Optional[List[AugmentationBlock]] = None):
        self.blocks = blocks if blocks is not None else []

    def add(self, block: AugmentationBlock) -> 'AugmentationPipeline':
        self.blocks.append(block)
        return self

    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """
        Apply all blocks in sequence.

        Returns augmented signal after all blocks
        """
        result = signal.copy()
        for block in self.blocks:
            result = block(result, fs, **kwargs)
        return result

    def __call__(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        return self.apply(signal, fs, **kwargs)

    def __repr__(self) -> str:
        block_names = [repr(b) for b in self.blocks]
        return f"AugmentationPipeline([{', '.join(block_names)}])"


# Additive White Gaussian Noise
class AWGNAugmentation(AugmentationBlock):
    def __init__(
        self,
        snr_db: float = 20.0,
        seed: Optional[int] = None
    ):
        """
            snr_db: Fixed SNR in dB (used if snr_range is None)
        """
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, _fs: float, **_kwargs) -> np.ndarray:
        """
        Add AWGN to the signal.

        Args:
            signal: Input signal (real or complex)
            _fs: Sample rate (not used for amplitude augmentation)

        Returns:
            Signal with added noise at specified SNR
        """
        # Determine SNR to use
        snr_db = self.snr_db
        
        sig_power = np.mean(np.abs(signal) ** 2)

        snr_linear = 10 ** (snr_db / 10)
        noise_power = sig_power / snr_linear

        # Generate noise based on signal type
        if np.iscomplexobj(signal):
            # Complex: noise power split between I and Q
            noise_std = np.sqrt(noise_power / 2)
            noise = (self.rng.normal(0, noise_std, signal.shape) +
                     1j * self.rng.normal(0, noise_std, signal.shape))
        else:
            noise_std = np.sqrt(noise_power)
            noise = self.rng.normal(0, noise_std, signal.shape)

        return signal + noise

    def __repr__(self) -> str:
        return f"AWGNAugmentation(snr_db={self.snr_db})"

class ScalarAmplitudeAndPhaseShift(AugmentationBlock):
    '''
    This block allows you to modulate the amplitude and/or the phase of a signal by
    multiplying the signal by a complex scalar

    amplitude: scalar to amplify or attenuate magnitude of signal
    phi: radians to delay signal by
    '''


    def __init__(
        self,
        amplitude: float = 1.0,
        phi: float = 0.0,
    ):
        self.amplitude = amplitude
        self.phi = phi

    def apply(self, signal: np.ndarray, _fs: float, **_kwargs) -> np.ndarray:
        phi = self.phi

        # build the complex channel scalar  h = |a| * e^(jφ)
        h = self.amplitude * np.exp(1j * phi)

        # take hilbert transform to get x_a = x + jH{x}
        x = hilbert(signal)
        y_a = h * x

        # recover real signal
        y = np.real(y_a)

        return y

    def __repr__(self) -> str:
        return f"ScalarAmplitudeAndPhaseShift(amplitude={self.amplitude:.4f}, phi={self.phi:.4f})"