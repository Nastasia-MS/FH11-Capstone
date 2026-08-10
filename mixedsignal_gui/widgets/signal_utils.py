"""
Shared demodulation utilities for constellation / IQ extraction.
Used by waveform_plots.py and comparison_widget.py.
"""

import numpy as np
from scipy import signal


# The RRC design lives with the generators that build waveforms from it, so
# there is a single implementation shared by generation and demodulation.
# Re-exported here because this module's own demodulator uses it and callers
# have historically imported it from this path.
from mixedsignal_gui.backend.generators import rcosdesign  # noqa: F401


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
    sps = int(round(sps))
    span = int(span)

    # ------------------------------------------------------------------
    # Step 1 — down-convert to complex baseband
    # ------------------------------------------------------------------
    if np.iscomplexobj(data):
        complex_baseband = np.asarray(data).ravel()
    else:
        t = np.arange(len(data)) / fs
        bb_raw = data * np.exp(-1j * 2 * np.pi * fc * t)

        # Low-pass filter to remove the image at 2·fc before matched
        # filtering.  Use zero-phase (forward-backward) filtering so
        # the group delay of the LPF does not shift the symbol timing.
        sig_bw = (1.0 + alpha) * (fs / sps) / 2.0   # one-sided BW
        lpf_cut = min(max(sig_bw * 2.0, fs / sps), fs / 2.0 * 0.9)
        sos = signal.butter(5, lpf_cut, 'low', fs=fs, output='sos')
        complex_baseband = 2.0 * signal.sosfiltfilt(sos, bb_raw)

    # ------------------------------------------------------------------
    # Step 2 — matched filter / detection
    # ------------------------------------------------------------------
    if pulse_shape == 'rrc':
        h = rcosdesign(alpha, span, sps)
        rx_filtered = np.convolve(complex_baseband, h, mode='full')

        # The RX RRC matched-filter group delay.  The TX already
        # trimmed its own filter delay, so the only delay we need to
        # compensate is the RX filter's group delay = (len(h)-1)/2.
        rx_delay = (len(h) - 1) // 2          # = span * sps // 2

        rx_sampled = rx_filtered[rx_delay::sps]

        # Trim edge symbols affected by filter transients
        if nsymb is not None:
            skip = max(span // 2, 2)
            end_idx = min(len(rx_sampled), nsymb) - skip
            if end_idx > skip:
                rx_recovered = rx_sampled[skip:end_idx]
            else:
                rx_recovered = rx_sampled
        else:
            # No nsymb — just drop the first and last *span* symbols
            drop = span
            if len(rx_sampled) > 2 * drop:
                rx_recovered = rx_sampled[drop:-drop]
            else:
                rx_recovered = rx_sampled
    else:
        # Rectangular pulse — simple LPF + down-sample
        symbol_rate = fs / sps
        cutoff = symbol_rate / 2.0 * 0.8
        sos = signal.butter(6, cutoff, 'low', fs=fs, output='sos')
        if np.iscomplexobj(data):
            # Baseband input — needs its own 2× gain (passband path
            # already applied 2× above).
            filtered = 2.0 * signal.sosfilt(sos, complex_baseband)
        else:
            filtered = signal.sosfilt(sos, complex_baseband)
        offset = sps // 2
        rx_recovered = filtered[offset::sps]

    # ------------------------------------------------------------------
    # Step 3 — normalise to unit average power
    # ------------------------------------------------------------------
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
