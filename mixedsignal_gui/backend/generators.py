import numpy as np
from typing import Optional


class MATLABWaveformGenerator:

    def __init__(self, matlab_engine: Optional[object]):
        self.matlab_engine = matlab_engine
        self.last_metadata = {}

    def _matlab_to_numpy(self, data):
        if getattr(data, 'is_complex', False):
            real_part = np.array(data.real, dtype=np.float64).flatten()
            imag_part = np.array(data.imag, dtype=np.float64).flatten()
            return (real_part + 1j * imag_part).astype(np.complex128)

        return np.asarray(data).flatten()

    def generate(self, cfg) -> np.ndarray:
        if self.matlab_engine is None or not getattr(self.matlab_engine, 'is_available', lambda: False)():
            raise RuntimeError("MATLAB engine is not available. Install/configure MATLAB and the MATLAB Engine for Python, or start the engine before generating waveforms.")

        from mixedsignal_gui.backend.core import WaveformConfig

        eng = self.matlab_engine.eng
        self.last_metadata = {}

        # Modulations that return symbols as a second output
        symbol_mods = {"PAM", "QAM", "PSK"}
        has_symbols = cfg.modulation in symbol_mods

        if has_symbols:
            data, symbols = eng.waveform_generator(
                float(cfg.output_len),
                float(cfg.fs),
                float(cfg.Tsymb),
                float(cfg.fc),
                float(cfg.M),
                cfg.modulation,
                "alpha", float(cfg.alpha),
                "span", float(cfg.span),
                "pulse_shape", cfg.pulse_shape,
                "output_type", cfg.output_type,
                nargout=2,
            )
            self.last_metadata["baseband_symbols"] = self._matlab_to_numpy(symbols)
        else:
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
                "output_type", cfg.output_type,
                nargout=1,
            )

        return self._matlab_to_numpy(data)