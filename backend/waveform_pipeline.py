import numpy as np
from scipy import signal

from backend.core import Waveform


class WaveformPipeline:
    def __init__(self, matlab_engine):
        self.matlab_engine = matlab_engine

    def generate(self, *, fs, Tsymb, Nsymb, fc, M, modulation,
                 var, alpha, span, pulse_shape, output_type="passband"):
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
            output_type=output_type,
        )

        waveform.generate_data()
        data = waveform.get_data()
        sps = waveform.get_sps()

        T = len(data) / fs
        t = np.linspace(0, T, len(data))

        # Compute spectrum in Python (plotspec_gui is just fft+fftshift
        # and cannot accept complex numpy arrays via the MATLAB engine API)
        N = len(data)
        ft = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))

        return {
            "time": t,
            "signal": data,
            "freqs": freqs,
            "spectrum": np.abs(ft),
            "sps": sps
        }
