"""
ParameterRange — tiny Qt-free value container used by the channel tab and
the bulk augmentation worker.

A parameter can be either a *fixed* value or a *uniform* range [low, high].
At apply time the bulk worker (and the single-waveform path) calls
`sample(rng)` to draw the value that will actually be used on that entry,
and then records what was drawn via `to_metadata(sampled)` so the per-file
sidecar captures both the range definition and the exact realisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ParameterRange:
    """Either a fixed value or a uniform(low, high) sampling range.

    Exactly one of (fixed) or (low, high) is populated. Use the class
    methods `ParameterRange.fixed(...)` / `ParameterRange.uniform(...)`
    rather than constructing directly.
    """

    _fixed: Optional[float] = None
    _low: Optional[float] = None
    _high: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def fixed(cls, value: float) -> "ParameterRange":
        return cls(_fixed=float(value))

    @classmethod
    def uniform(cls, low: float, high: float) -> "ParameterRange":
        low_f = float(low)
        high_f = float(high)
        if high_f < low_f:
            low_f, high_f = high_f, low_f
        return cls(_low=low_f, _high=high_f)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def is_fixed(self) -> bool:
        return self._fixed is not None

    @property
    def fixed_value(self) -> float:
        if self._fixed is None:
            raise ValueError("ParameterRange is not fixed")
        return self._fixed

    @property
    def low(self) -> float:
        if self._low is None:
            raise ValueError("ParameterRange is not a range")
        return self._low

    @property
    def high(self) -> float:
        if self._high is None:
            raise ValueError("ParameterRange is not a range")
        return self._high

    # ------------------------------------------------------------------
    # Sampling / metadata
    # ------------------------------------------------------------------
    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Return a float realisation of this parameter.

        For fixed ranges this is the stored value. For uniform ranges a
        draw is taken from `rng.uniform(low, high)`; if `rng` is None a
        fresh default generator is used.
        """
        if self._fixed is not None:
            return float(self._fixed)
        if rng is None:
            rng = np.random.default_rng()
        return float(rng.uniform(self._low, self._high))

    def to_metadata(self, sampled: float) -> dict:
        """Sidecar-friendly description of the range and the value drawn."""
        if self._fixed is not None:
            return {"mode": "fixed", "value": float(self._fixed)}
        return {
            "mode": "uniform",
            "low": float(self._low),
            "high": float(self._high),
            "sampled": float(sampled),
        }
