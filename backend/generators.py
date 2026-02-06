import numpy as np
from typing import Optional


class MATLABWaveformGenerator:

    def __init__(self, matlab_engine: Optional[object]):
        self.matlab_engine = matlab_engine

    def generate(self, cfg) -> np.ndarray:
        if self.matlab_engine is None or not getattr(self.matlab_engine, 'is_available', lambda: False)():
            raise RuntimeError("MATLAB engine is not available. Install/configure MATLAB and the MATLAB Engine for Python, or start the engine before generating waveforms.")

        eng = self.matlab_engine.eng
        data = eng.waveform_generator(
            float(cfg.output_len),
            float(cfg.fs),
            float(cfg.Tsymb),
            float(cfg.fc),
            float(cfg.M),
            cfg.modulation,
            "alpha", float(cfg.alpha),
            "span", float(cfg.span),
            "pulse_shape", cfg.pulse_shape,
            nargout=1,
        )
        return np.asarray(data).flatten()