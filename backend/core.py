from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class WaveformConfig:
    modulation: str
    fs: float
    Tsymb: float
    fc: float
    M: int
    Nsymb: int

    var: Optional[float] = None
    freq_sep: Optional[float] = None

    alpha: float = 0.35
    span: int = 8
    pulse_shape: str = "rrc"
    output_type: str = "passband"

    def __post_init__(self):
        if isinstance(self.M, float) and self.M.is_integer():
            self.M = int(self.M)

        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")

        if self.pulse_shape not in {"rrc", "rect"}:
            raise ValueError("pulse_shape must be 'rrc' or 'rect'")

        if self.span < 1:
            raise ValueError("span must be >= 1")

        self._validate()

    def _validate(self):
        sps = self.fs * self.Tsymb
        if abs(sps - round(sps)) > 1e-9:
            raise ValueError("fs * Tsymb must be an integer")

        if self.output_type == "passband" and self.fc >= self.fs / 2:
            raise ValueError("fc must be < fs/2")

        if self.modulation == "FSK":
            self._require_power_of_2()
        elif self.modulation == "QAM":
            self._require_power_of_2(min_val=4)
        elif self.modulation == "PAM":
            if self.M < 2:
                raise ValueError("PAM requires M >= 2")
        elif self.modulation == "PSK":
            self._require_power_of_2()
        elif self.modulation == "FHSS":
            if self.M < 2:
                raise ValueError("FHSS requires M >= 2")
        else:
            raise ValueError(f"Unknown modulation: {self.modulation}")

    def _require_power_of_2(self, min_val=2):
        if self.M < min_val or (self.M & (self.M - 1)) != 0:
            raise ValueError(f"{self.modulation} requires M to be power of 2 >= {min_val}")

    @property
    def sps(self) -> int:
        return int(self.fs * self.Tsymb)

    @property
    def output_len(self) -> int:
        return self.sps * self.Nsymb


class Waveform:

    def __init__(
        self,
        *,
        fs: float,
        Tsymb: float,
        Nsymb: int,
        fc: float,
        M: int,
        modulation: str,
        matlab_engine=None,
        generator_impl=None,
        var: Optional[float] = None,
        freq_sep: Optional[float] = None,
        alpha: float = 0.35,
        span: int = 8,
        pulse_shape: str = "rrc",
        output_type: str = "passband",
    ):
        self.config = WaveformConfig(
            modulation=modulation,
            fs=fs,
            Tsymb=Tsymb,
            fc=fc,
            M=M,
            Nsymb=Nsymb,
            var=var,
            freq_sep=freq_sep,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            output_type=output_type,
        )

        self.matlab_engine = matlab_engine
        self._generator = generator_impl
        self._data: Optional[np.ndarray] = None

    def _ensure_generator(self):
        if self._generator is not None:
            return
        from backend.generators import MATLABWaveformGenerator

        self._generator = MATLABWaveformGenerator(self.matlab_engine)

    def generate(self) -> np.ndarray:
        self._ensure_generator()
        self._data = self._generator.generate(self.config)
        return self._data

    def generate_data(self):
        self.generate()
        return self._data

    def get_data(self):
        if self._data is None:
            raise ValueError("Call generate_data() first")
        return self._data

    def get_sps(self):
        return self.config.sps

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("Waveform not generated yet")
        return self._data
