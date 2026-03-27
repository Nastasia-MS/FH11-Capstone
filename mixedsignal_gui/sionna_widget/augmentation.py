"""Channel augmentation block and parameters dataclass.

Provides both a legacy FIR tapped-delay-line model (``apply``) and a
deterministic RT channel model using Sionna's ``ApplyTimeChannel``
(``apply_with_taps``).  Both conform to the
``AugmentationBlock.apply(signal, fs, **kwargs)`` interface.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ChannelParameters:
    """Raw channel parameters extracted from Sionna ray tracing."""

    delays: np.ndarray = field(default_factory=lambda: np.array([]))
    complex_gains: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    powers_linear: np.ndarray = field(default_factory=lambda: np.array([]))

    aod_theta: Optional[np.ndarray] = None
    aod_phi: Optional[np.ndarray] = None
    aoa_theta: Optional[np.ndarray] = None
    aoa_phi: Optional[np.ndarray] = None

    frequency_hz: float = 3.5e9
    num_paths: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for export."""
        d = {
            "delays": self.delays.tolist() if self.delays.size else [],
            "complex_gains_real": self.complex_gains.real.tolist() if self.complex_gains.size else [],
            "complex_gains_imag": self.complex_gains.imag.tolist() if self.complex_gains.size else [],
            "powers_linear": self.powers_linear.tolist() if self.powers_linear.size else [],
            "frequency_hz": self.frequency_hz,
            "num_paths": self.num_paths,
        }
        for name in ("aod_theta", "aod_phi", "aoa_theta", "aoa_phi"):
            arr = getattr(self, name)
            d[name] = arr.tolist() if arr is not None and arr.size else None
        return d


class SionnaChannelAugmentation:
    """Multipath channel augmentation using a FIR tapped-delay-line model.

    Conforms to AugmentationBlock.apply(signal, fs, **kwargs) via duck-typing.
    No subclassing required -- the caller's pipeline just needs ``apply()``.

    The model converts ray-tracing path data into baseband-equivalent taps::

        a_bb[i] = a[i] * exp(-j * 2pi * fc * tau[i])
        sample_index[i] = round(tau[i] * fs)

    Taps that map to the same sample index are summed.  The resulting sparse
    FIR filter is convolved with the input signal.
    """

    def __init__(self):
        self._params: Optional[ChannelParameters] = None

    # -- public API --------------------------------------------------------

    def update_params(self, params: ChannelParameters):
        """Refresh the channel model after a new simulation run."""
        self._params = params

    @property
    def params(self) -> Optional[ChannelParameters]:
        return self._params

    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """Apply multipath channel to *signal* sampled at *fs* Hz.

        Returns a copy of *signal* (same length) if no parameters are set.
        """
        if self._params is None or self._params.num_paths == 0:
            return signal.copy()

        h = self._build_fir(fs)
        out = np.convolve(signal, h)
        # Trim to input length (causal filter, keep head)
        return out[: len(signal)]

    def get_impulse_response(self, fs: float, N: int) -> np.ndarray:
        """Return the length-*N* impulse response at sample rate *fs*."""
        if self._params is None or self._params.num_paths == 0:
            ir = np.zeros(N, dtype=np.complex128)
            if N > 0:
                ir[0] = 1.0
            return ir

        h = self._build_fir(fs)
        ir = np.zeros(N, dtype=np.complex128)
        ir[: min(len(h), N)] = h[: min(len(h), N)]
        return ir

    # -- internals ---------------------------------------------------------

    def apply_with_taps(
        self,
        signal: np.ndarray,
        fs: float,
        taps: np.ndarray,
        waveform_length: int,
        tx_power_dbm: float,
        noise_power_dbm: float,
        num_tx_ant: int,
        rx_antenna_index: int = 0,
    ) -> np.ndarray:
        """Apply deterministic RT channel using Sionna's ApplyTimeChannel.

        Args:
            signal: Input complex64 signal of shape (N,).
            fs: Sample rate in Hz (unused but kept for interface compat).
            taps: Channel taps, shape (num_rx, num_tx, num_rx_ant, num_tx_ant, L_TOT).
            waveform_length: Desired time dimension T for the channel.
            tx_power_dbm: Transmit power in dBm.
            noise_power_dbm: Noise power in dBm.
            num_tx_ant: Number of TX antenna elements.
            rx_antenna_index: Which RX antenna element to select from output.

        Returns:
            Complex64 array of shape (waveform_length,).
        """
        import tensorflow as tf
        from sionna.phy.channel import ApplyTimeChannel

        x = np.asarray(signal, dtype=np.complex64).reshape(-1)

        # Normalize to unit power
        p = np.mean(np.abs(x) ** 2)
        if p > 0:
            x = x / np.sqrt(p)

        # Scale by TX power (dBm -> watts -> amplitude)
        p_lin = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
        x = x * np.sqrt(p_lin)

        # Pad or trim to waveform_length
        T = waveform_length
        if len(x) < T:
            x = np.pad(x, (0, T - len(x)), mode="constant", constant_values=0)
        else:
            x = x[:T]

        # Replicate across TX antennas: (1, 1, num_tx_ant, T)
        input_wave = np.stack([x] * num_tx_ant, axis=0)  # (num_tx_ant, T)
        input_wave = input_wave[np.newaxis, np.newaxis, :, :]  # (1, 1, num_tx_ant, T)

        L_TOT = taps.shape[-1]

        # Noise: dBm -> watts
        noise_linear = 10.0 ** (noise_power_dbm / 10.0) * 1e-3

        apply_ch = ApplyTimeChannel(num_time_samples=T, l_tot=L_TOT)
        output = apply_ch(
            tf.constant(input_wave, dtype=tf.complex64),
            tf.constant(taps, dtype=tf.complex64),
            tf.constant(noise_linear, dtype=tf.float32),
        )
        # output shape: (1, 1, num_rx_ant, T)
        out_np = output.numpy()
        return out_np[0, 0, rx_antenna_index, :].astype(np.complex64)

    # -- internals ---------------------------------------------------------

    def _build_fir(self, fs: float) -> np.ndarray:
        """Build the baseband FIR tap vector."""
        p = self._params
        fc = p.frequency_hz

        # Baseband equivalent gains
        a_bb = p.complex_gains * np.exp(-1j * 2 * np.pi * fc * p.delays)

        # Quantize delays to sample indices
        indices = np.rint(p.delays * fs).astype(int)
        indices = np.clip(indices, 0, None)

        length = int(indices.max()) + 1 if indices.size else 1
        h = np.zeros(length, dtype=np.complex128)
        for idx, gain in zip(indices, a_bb):
            h[idx] += gain

        return h
