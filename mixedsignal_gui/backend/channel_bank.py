"""Bank of measured channel impulse responses loaded from disk.

Gives the Channel & Noise tab a third source of channel realisations
alongside the stochastic TDL model and the Sionna ray tracer: real CIRs
recorded from hardware.

Files are expected to contain impulse responses, not raw captures — a
complex tap vector h[n] per measurement.  Supported containers:

* ``.npy``   NumPy array.  1-D is one CIR; 2-D is one CIR per row.
* ``.mat``   MATLAB v5 (via scipy) or v7.3/HDF5 (via h5py).  The largest
             complex variable is used unless a key is given.
* ``.sigmf`` SigMF recording: a ``.sigmf-meta`` JSON sidecar naming the
             sample format, beside a ``.sigmf-data`` blob.  Parsed directly
             so the optional ``sigmf`` package is not required.
* ``.bin``   Headerless raw samples.  The layout cannot be inferred, so the
             caller supplies a dtype; see ``BIN_DTYPES``.

Everything is normalised to complex128 taps so downstream convolution has a
single case to handle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scipy.io as sio
except ImportError:                                   # pragma: no cover
    sio = None

try:
    import h5py
except ImportError:                                   # pragma: no cover
    h5py = None


#: Sample layouts for headerless ``.bin`` files, shown in the import dialog.
#: complex64 (interleaved float32 I/Q) is the usual SDR convention.
BIN_DTYPES = {
    "complex64  (2x float32 I/Q)": np.complex64,
    "complex128 (2x float64 I/Q)": np.complex128,
    "float32    (real)":           np.float32,
    "int16      (interleaved I/Q)": np.int16,
}

SUPPORTED_SUFFIXES = (".npy", ".mat", ".sigmf", ".sigmf-meta", ".sigmf-data", ".bin")

#: Longer than this and a "CIR" is almost certainly a raw capture; loading it
#: as a filter would produce a convolution nobody wants to wait for.
MAX_REASONABLE_TAPS = 100_000


@dataclass
class MeasuredChannel:
    """One channel impulse response plus where it came from."""

    name: str
    taps: np.ndarray                 # complex128, 1-D
    source_path: str = ""
    fs: Optional[float] = None       # sample rate in Hz, when the file says
    metadata: dict = field(default_factory=dict)

    @property
    def num_taps(self) -> int:
        return len(self.taps)

    @property
    def delay_spread_samples(self) -> float:
        """RMS delay spread in samples — the usual one-number summary of a CIR."""
        power = np.abs(self.taps) ** 2
        total = power.sum()
        if total <= 0:
            return 0.0
        n = np.arange(len(power))
        mean_delay = float((n * power).sum() / total)
        return float(np.sqrt(((n - mean_delay) ** 2 * power).sum() / total))

    def normalized(self) -> np.ndarray:
        """Taps scaled to unit energy, so applying a channel preserves signal power."""
        energy = np.sqrt(np.sum(np.abs(self.taps) ** 2))
        return self.taps / energy if energy > 0 else self.taps

    def describe(self) -> str:
        fs_txt = f", fs={self.fs/1e6:.3g} MHz" if self.fs else ""
        return (f"{self.num_taps} taps, RMS delay spread "
                f"{self.delay_spread_samples:.1f} samples{fs_txt}")


# ── loaders ──────────────────────────────────────────────────────────────

def _as_cir_list(arr: np.ndarray, stem: str, path: str, fs=None,
                 metadata=None) -> list[MeasuredChannel]:
    """Turn a loaded array into one or more channels.

    1-D is a single CIR.  2-D is a set of measurements, one per row — the
    array is transposed first if it looks column-major (far more columns than
    rows is the giveaway for taps-along-axis-0).
    """
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 0:
        raise ValueError("file contains a scalar, not an impulse response")

    if not np.iscomplexobj(arr):
        # Real taps are legitimate (a real-valued channel), so allow them.
        arr = arr.astype(np.float64)

    if arr.ndim == 1:
        return [MeasuredChannel(stem, arr.astype(np.complex128), path, fs,
                                metadata or {})]

    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])

    # Prefer rows = measurements, columns = taps.
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T

    return [
        MeasuredChannel(f"{stem}[{i}]", row.astype(np.complex128), path, fs,
                        metadata or {})
        for i, row in enumerate(arr)
    ]


def _load_npy(path: str) -> list[MeasuredChannel]:
    return _as_cir_list(np.load(path, allow_pickle=False), Path(path).stem, path)


def _load_mat(path: str, mat_key: Optional[str] = None) -> list[MeasuredChannel]:
    """Load a MATLAB file, transparently handling v7.3 (HDF5) files."""
    try:
        md = sio.loadmat(path) if sio is not None else None
        if md is None:
            raise RuntimeError("scipy is required to read .mat files")
        candidates = {k: v for k, v in md.items() if not k.startswith("__")}
    except NotImplementedError:
        # v7.3 files are HDF5 and scipy refuses them
        if h5py is None:
            raise RuntimeError(
                "This looks like a MATLAB v7.3 file, which needs h5py.") from None
        with h5py.File(path, "r") as f:
            candidates = {}
            for k in f.keys():
                v = np.array(f[k])
                # MATLAB writes complex HDF5 as a compound dtype
                if v.dtype.names and set(v.dtype.names) >= {"real", "imag"}:
                    v = v["real"] + 1j * v["imag"]
                candidates[k] = v

    if mat_key is not None:
        if mat_key not in candidates:
            raise ValueError(f"'{mat_key}' not found in {os.path.basename(path)}; "
                             f"available: {sorted(candidates)}")
        chosen, arr = mat_key, candidates[mat_key]
    else:
        arrays = {k: np.asarray(v) for k, v in candidates.items()
                  if isinstance(v, np.ndarray) and v.size > 1}
        if not arrays:
            raise ValueError(f"no array variables in {os.path.basename(path)}")
        # A complex variable is almost certainly the CIR; otherwise take the
        # largest, which beats picking whichever happens to be first.
        complex_arrays = {k: v for k, v in arrays.items() if np.iscomplexobj(v)}
        pool = complex_arrays or arrays
        chosen = max(pool, key=lambda k: pool[k].size)
        arr = pool[chosen]

    return _as_cir_list(arr, f"{Path(path).stem}:{chosen}", path,
                        metadata={"mat_key": chosen})


#: SigMF core:datatype -> numpy dtype.  Only little-endian is handled; the
#: spec allows big-endian ("_be") too, which is flagged rather than guessed.
_SIGMF_DTYPES = {
    "cf32": np.complex64, "cf64": np.complex128,
    "ci16": np.int16, "ci8": np.int8,
    "rf32": np.float32, "rf64": np.float64,
    "ri16": np.int16, "ri8": np.int8,
}


def _load_sigmf(path: str) -> list[MeasuredChannel]:
    """Load a SigMF recording from its metadata sidecar.

    Parsed directly rather than via the ``sigmf`` package so importing a
    recording does not add a dependency.
    """
    p = Path(path)
    if p.suffix == ".sigmf-data":
        meta_path = p.with_suffix(".sigmf-meta")
    elif p.suffix in (".sigmf-meta", ".sigmf"):
        meta_path = p.with_suffix(".sigmf-meta")
    else:
        meta_path = Path(str(p) + ".sigmf-meta")

    if not meta_path.is_file():
        raise ValueError(f"no SigMF metadata found at {meta_path.name}")

    meta = json.loads(meta_path.read_text())
    global_meta = meta.get("global", {})
    datatype = global_meta.get("core:datatype", "")

    base = datatype[:-3] if datatype.endswith("_le") else datatype
    if datatype.endswith("_be"):
        raise ValueError(f"big-endian SigMF ({datatype}) is not supported")
    if base not in _SIGMF_DTYPES:
        raise ValueError(f"unsupported SigMF datatype '{datatype}'")

    data_path = meta_path.with_suffix(".sigmf-data")
    if not data_path.is_file():
        raise ValueError(f"SigMF data file missing: {data_path.name}")

    raw = np.fromfile(data_path, dtype=_SIGMF_DTYPES[base])
    if base.startswith("ci"):          # interleaved integer I/Q
        raw = raw.astype(np.float64)
        raw = raw[0::2] + 1j * raw[1::2]

    fs = global_meta.get("core:sample_rate")
    return _as_cir_list(raw, meta_path.stem, str(data_path),
                        fs=float(fs) if fs else None,
                        metadata={"sigmf": global_meta})


def _load_bin(path: str, dtype=np.complex64) -> list[MeasuredChannel]:
    """Load headerless samples using a caller-supplied dtype."""
    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError("file is empty or the chosen sample format does not fit it")
    if dtype == np.int16:              # interleaved I/Q
        if raw.size % 2:
            raw = raw[:-1]
        raw = raw.astype(np.float64)
        raw = raw[0::2] + 1j * raw[1::2]
    return _as_cir_list(raw, Path(path).stem, path,
                        metadata={"bin_dtype": np.dtype(dtype).name})


def load_channel_file(path: str, *, bin_dtype=np.complex64,
                      mat_key: Optional[str] = None) -> list[MeasuredChannel]:
    """Load one file into a list of channels. Raises ValueError on bad input."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        channels = _load_npy(path)
    elif suffix == ".mat":
        channels = _load_mat(path, mat_key)
    elif suffix in (".sigmf", ".sigmf-meta", ".sigmf-data"):
        channels = _load_sigmf(path)
    elif suffix == ".bin":
        channels = _load_bin(path, bin_dtype)
    else:
        raise ValueError(f"unsupported file type '{suffix}'")

    for ch in channels:
        if ch.num_taps == 0:
            raise ValueError(f"{ch.name} has no taps")
        if ch.num_taps > MAX_REASONABLE_TAPS:
            raise ValueError(
                f"{ch.name} has {ch.num_taps:,} taps, which looks like a raw "
                f"capture rather than an impulse response. This bank expects "
                f"CIRs (typically tens to thousands of taps).")
    return channels


# ── the bank ─────────────────────────────────────────────────────────────

class ChannelBank:
    """Collection of measured channels, keyed by unique name."""

    def __init__(self):
        self._channels: list[MeasuredChannel] = []

    def __len__(self) -> int:
        return len(self._channels)

    def __iter__(self):
        return iter(self._channels)

    @property
    def channels(self) -> list[MeasuredChannel]:
        return list(self._channels)

    def names(self) -> list[str]:
        return [c.name for c in self._channels]

    def get(self, name: str) -> Optional[MeasuredChannel]:
        for c in self._channels:
            if c.name == name:
                return c
        return None

    def clear(self):
        self._channels.clear()

    def add(self, channel: MeasuredChannel):
        """Add a channel, de-duplicating its name so the list stays selectable."""
        existing = set(self.names())
        if channel.name in existing:
            base, n = channel.name, 2
            while f"{base} ({n})" in existing:
                n += 1
            channel.name = f"{base} ({n})"
        self._channels.append(channel)

    def add_file(self, path: str, **kwargs) -> int:
        """Load one file into the bank. Returns how many channels were added."""
        channels = load_channel_file(path, **kwargs)
        for ch in channels:
            self.add(ch)
        return len(channels)

    def add_folder(self, folder: str, **kwargs) -> tuple[int, dict]:
        """Load every supported file in *folder* (non-recursive).

        Returns ``(channels_added, {filename: error})`` — one bad file does not
        abort the import, since a folder of recordings often has strays in it.
        """
        added, errors = 0, {}
        entries = sorted(Path(folder).iterdir())
        for entry in entries:
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            # .sigmf-data is loaded via its own .sigmf-meta; skip to avoid dupes
            if suffix == ".sigmf-data":
                continue
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            try:
                added += self.add_file(str(entry), **kwargs)
            except Exception as exc:                       # noqa: BLE001
                errors[entry.name] = str(exc)
        return added, errors
