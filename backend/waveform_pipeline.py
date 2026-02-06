import numpy as np
from scipy import signal

from backend.core import Waveform


class WaveformPipeline:
    def __init__(self, matlab_engine):
        self.matlab_engine = matlab_engine

    def generate(self, *, fs, Tsymb, Nsymb, fc, M, modulation,
                 var, alpha, span, pulse_shape):
        waveform = Waveform(
            fs=fs,
            Tsymb=Tsymb,
            Nsymb=Nsymb,
            fc=fc,
            M=M,
            modulation=modulation,
            var=var,
            matlab_engine=self.matlab_engine,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
        )

        waveform.generate_data()
        data = waveform.get_data()
        sps = waveform.get_sps()

        T = len(data) / fs
        t = np.linspace(0, T, len(data))

        if self.matlab_engine is None or not getattr(self.matlab_engine, 'is_available', lambda: False)():
            raise RuntimeError(
                "MATLAB engine unavailable — waveform generation and spectrum analysis require MATLAB."
            )

        try:
            freqs, ft = self.matlab_engine.eng.plotspec_gui(data, 1 / fs, nargout=2)
            freqs = np.array(freqs).flatten()
            ft = np.array(ft).flatten()
        except Exception as e:
            raise RuntimeError(f"Failed to run MATLAB plotspec_gui: {e}") from e

        return {
            "time": t,
            "signal": data,
            "freqs": freqs,
            "spectrum": np.abs(ft),
            "sps": sps
        }
