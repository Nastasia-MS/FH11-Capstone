"""Numerically compare the Python waveform generator against MATLAB.

The Python generator in ``generators.py`` was written by porting
``waveform_functions/waveform_generator.m``, but it was developed on a machine
without MATLAB, so it has only ever been checked against theory (constellation
geometry, filter energy, tone spacing, Barker autocorrelation) — never against
the reference implementation itself.

Run this wherever MATLAB and the MATLAB Engine for Python are installed:

    python -m mixedsignal_gui.backend.compare_generators

Both engines are driven through the same ``WaveformConfig``, so any difference
is a real divergence rather than a difference in how they were called.

What to expect:

* PAM / QAM / PSK — the symbol *sequences* will differ, because each engine
  draws its own random data and the Python side deliberately does not
  reproduce Gray coding (any bijection onto the constellation is equivalent
  for uniform random data).  Compare the statistics reported here, not
  sample-by-sample values: power, spectrum shape, constellation geometry.
* FSK / FHSS — same, plus a random symbol stream.
* LFM / Barker / FMCW — MATLAB builds these with the Phased Array System
  Toolbox; the Python versions are independent reimplementations of the same
  underlying maths, so the envelopes should agree in character but not
  necessarily sample for sample.
* WiFi / LTE / 5G_NR — Python refuses these outright; only MATLAB can build
  standards-compliant frames.

A known divergence: MATLAB's ``qammod`` uses a *cross* constellation for
M = 32 and 128, while this port uses a rectangular one.  Square M (4, 16, 64 —
the defaults everywhere in the app) and M = 8 agree.
"""

import numpy as np

from mixedsignal_gui.backend.core import WaveformConfig
from mixedsignal_gui.backend.generators import (
    MATLABWaveformGenerator, PythonWaveformGenerator, MATLAB_ONLY_MODULATIONS,
)

# (modulation, M) pairs worth comparing, with the defaults the app ships.
CASES = [
    ("PAM", 2), ("PAM", 4), ("PAM", 8),
    ("QAM", 4), ("QAM", 16), ("QAM", 64),
    ("PSK", 2), ("PSK", 4), ("PSK", 8),
    ("FSK", 2), ("FSK", 4),
    ("FHSS", 4), ("FHSS", 8),
    ("LFM", 4), ("Barker", 13), ("FMCW", 4),
]


def _stats(sig, fs):
    """Engine-comparable summary of a waveform (order-independent)."""
    sig = np.asarray(sig).ravel()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    total = spec.sum() or 1.0
    centroid = float((freqs * spec).sum() / total)
    return {
        "len": len(sig),
        "complex": bool(np.iscomplexobj(sig)),
        "mean_power": float(np.mean(np.abs(sig) ** 2)),
        "peak": float(np.max(np.abs(sig))),
        "peak_freq_hz": float(freqs[int(np.argmax(spec))]),
        "spec_centroid_hz": centroid,
    }


def compare(matlab_engine, cases=CASES, *, fs=100e3, Tsymb=1e-3, fc=10e3,
            Nsymb=128, output_type="passband", pulse_shape="rrc"):
    """Print a side-by-side comparison. Returns True if nothing looked wrong."""
    mat = MATLABWaveformGenerator(matlab_engine)
    pyg = PythonWaveformGenerator(seed=0)

    print(f"{'case':<12} {'field':<18} {'MATLAB':>14} {'Python':>14}  {'':<6}")
    print("-" * 70)
    all_ok = True

    for modulation, M in cases:
        if modulation in MATLAB_ONLY_MODULATIONS:
            continue
        try:
            cfg = WaveformConfig(
                modulation=modulation, fs=fs, Tsymb=Tsymb, fc=fc, M=M,
                Nsymb=Nsymb, pulse_shape=pulse_shape, output_type=output_type)
            m_stats = _stats(mat.generate(cfg), fs)
            p_stats = _stats(pyg.generate(cfg), fs)
        except Exception as exc:                       # noqa: BLE001
            print(f"{modulation}-{M:<9} ERROR: {exc}")
            all_ok = False
            continue

        label = f"{modulation}-{M}"
        for key in ("len", "complex", "mean_power", "peak",
                    "peak_freq_hz", "spec_centroid_hz"):
            m_val, p_val = m_stats[key], p_stats[key]
            if isinstance(m_val, bool) or isinstance(m_val, int):
                match = m_val == p_val
                fmt = lambda v: str(v)                 # noqa: E731
            else:
                # Random data means exact equality is not expected; flag only
                # differences too large to be sampling noise.
                denom = max(abs(m_val), abs(p_val), 1e-12)
                match = abs(m_val - p_val) / denom < 0.15
                fmt = lambda v: f"{v:.4g}"             # noqa: E731
            mark = "" if match else "  <-- DIFFERS"
            if not match:
                all_ok = False
            print(f"{label:<12} {key:<18} {fmt(m_val):>14} {fmt(p_val):>14}{mark}")
            label = ""
        print()

    print("Everything within tolerance." if all_ok else
          "Differences flagged above — investigate before trusting either engine.")
    return all_ok


def main():
    from mixedsignal_gui.backend.matlab_engine import MatlabEngine

    engine = MatlabEngine(lazy=False)
    if not engine.is_available():
        raise SystemExit(
            "MATLAB engine unavailable — this script exists specifically to "
            "compare against MATLAB, so there is nothing to do without it.")

    import os
    engine.add_path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "waveform_functions"))
    compare(engine)


if __name__ == "__main__":
    main()
