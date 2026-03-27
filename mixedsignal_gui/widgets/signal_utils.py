"""
Shared demodulation utilities for constellation / IQ extraction.
Used by waveform_plots.py and comparison_widget.py.
"""

import numpy as np
from scipy import signal


def rcosdesign(alpha, span, sps):
    """
    Pure-Python root raised cosine (RRC) filter design.

    Equivalent to MATLAB's ``rcosdesign(alpha, span, sps, 'sqrt')``.

    Returns a 1-D unit-energy filter array of length ``span * sps + 1``.
    """
    N = span * sps
    n = np.arange(-N // 2, N // 2 + 1, dtype=float)
    t = n / sps  # normalised time in symbol periods

    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + 4.0 * alpha / np.pi
        elif alpha != 0.0 and abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-12:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            if abs(den) < 1e-30:
                h[i] = 0.0
            else:
                h[i] = num / den

    energy = np.sqrt(np.sum(h ** 2))
    if energy > 0:
        h = h / energy

    return h


def demodulate_to_symbols(data, fs, fc, sps, alpha=0.35, span=8,
                          pulse_shape='rrc', nsymb=None):
    """
    Extract constellation symbols from a signal.

    Parameters
    ----------
    data : array
        Complex baseband or real passband signal.
    fs : float
        Sampling frequency (Hz).
    fc : float or None
        Carrier frequency (Hz). Ignored for complex input.
    sps : int
        Samples per symbol.
    alpha : float
        RRC roll-off factor.
    span : int
        RRC filter span in symbols.
    pulse_shape : str
        ``'rrc'`` or ``'rect'``.
    nsymb : int or None
        Number of data symbols (for transient trimming).

    Returns
    -------
    symbols : 1-D complex ndarray
        Demodulated, normalised symbol array.
    """
    sps = int(sps)
    span = int(span)

    if np.iscomplexobj(data):
        complex_baseband = data
    else:
        t = np.arange(len(data)) / fs
        complex_baseband = data * np.exp(-1j * 2 * np.pi * fc * t)

    if pulse_shape == 'rrc':
        h = rcosdesign(alpha, span, sps)
        rx_filtered = np.convolve(complex_baseband, h, mode='full')
        rx_filtered = 2.0 * rx_filtered

        total_delay = span * sps
        rx_sampled = rx_filtered[total_delay::sps]

        if nsymb is not None:
            skip = int(span)
            start = skip
            end = int(min(nsymb - skip, len(rx_sampled) - skip))
            if end > start:
                rx_recovered = rx_sampled[start:end]
            else:
                rx_recovered = rx_sampled[:int(nsymb)]
        else:
            rx_recovered = rx_sampled
    else:
        symbol_rate = fs / sps
        cutoff = symbol_rate / 2.0 * 0.8
        sos = signal.butter(6, cutoff, 'low', fs=fs, output='sos')
        filtered = 2.0 * signal.sosfilt(sos, complex_baseband)
        offset = sps // 2
        rx_recovered = filtered[offset::sps]

    avg_power = np.mean(np.abs(rx_recovered) ** 2)
    if avg_power > 0:
        rx_recovered = rx_recovered / np.sqrt(avg_power)

    return rx_recovered


def extract_fsk_iq(data, fs, fc, sps, M):
    """
    Extract FSK IQ trajectory (for scatter / trajectory plots).

    Returns
    -------
    I, Q : 1-D ndarrays  (down-sampled for visualisation)
    """
    M = int(M)
    sps = int(sps)
    Tsymb = sps / fs
    freq_sep = 1.0 / Tsymb

    if np.iscomplexobj(data):
        # Baseband FSK: tones span ±(M-1)/2 * freq_sep around DC.
        # Use a bandpass-like approach: no filtering needed for IQ trajectory
        # since the signal is already baseband.  Just downsample for display.
        complex_baseband = data
    else:
        t = np.arange(len(data)) / fs
        complex_baseband = data * np.exp(-1j * 2 * np.pi * fc * t)

        cutoff = min(M * freq_sep, fs / 2 * 0.9)
        sos = signal.butter(4, cutoff, 'low', fs=fs, output='sos')
        complex_baseband = 2.0 * signal.sosfilt(sos, complex_baseband)

    ds = max(1, sps // 10)
    I = np.real(complex_baseband[::ds])
    Q = np.imag(complex_baseband[::ds])
    return I, Q


def extract_fhss_iq(data, fs, fc, sps, M):
    """
    Extract FHSS IQ trajectory.

    Returns
    -------
    I, Q : 1-D ndarrays  (down-sampled for visualisation)
    """
    M = int(M)
    sps = int(sps)
    channel_spacing = fs / (2 * M)
    hop_bw = channel_spacing * (M - 1)

    if np.iscomplexobj(data):
        # Baseband FHSS: hops span the baseband directly.
        # No filtering needed — just downsample for display.
        complex_baseband = data
    else:
        t = np.arange(len(data)) / fs
        complex_baseband = data * np.exp(-1j * 2 * np.pi * fc * t)

        cutoff = min(hop_bw * 1.5, fs / 2 * 0.9)
        sos = signal.butter(4, cutoff, 'low', fs=fs, output='sos')
        complex_baseband = 2.0 * signal.sosfilt(sos, complex_baseband)

    ds = max(1, sps // 10)
    I = np.real(complex_baseband[::ds])
    Q = np.imag(complex_baseband[::ds])
    return I, Q
