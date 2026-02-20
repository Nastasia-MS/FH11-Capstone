"""Background worker for running Sionna path computation off the main thread."""

from PySide6.QtCore import QObject, Signal, Slot


class PathWorker(QObject):
    """Runs engine.compute_paths() on a QThread."""

    paths_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @Slot()
    def compute_paths(self):
        try:
            result = self.engine.compute_paths()
            if result:
                self.paths_ready.emit(result)
            else:
                self.error.emit("Path computation returned no results")
        except Exception as e:
            self.error.emit(str(e))
