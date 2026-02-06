import os
import traceback

try:
    import matlab.engine
except Exception:
    matlab = None


class MatlabEngine:

    def __init__(self, lazy: bool = True):
        self.eng = None
        self._started = False
        self.lazy = lazy
        if not self.lazy:
            self.start()

    def start(self):
        if self._started and self.eng is not None:
            return
        if matlab is None:
            print("MATLAB Engine for Python not available (import failed).")
            self.eng = None
            self._started = False
            return

        try:
            self.eng = matlab.engine.start_matlab()
            self._started = True
            print("MATLAB engine started")
        except Exception as e:
            self.eng = None
            self._started = False
            print("Failed to start MATLAB engine:", str(e))

    def is_available(self) -> bool:
        return self.eng is not None

    def add_path(self, path: str):
        if self.eng is None:
            print(f"Cannot add MATLAB path; engine not available. Tried: {path}")
            return
        try:
            self.eng.addpath(path, nargout=0)
        except Exception as e:
            print(f"Failed to add path {path} to MATLAB: {e}")
