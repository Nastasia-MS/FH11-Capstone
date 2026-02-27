# Mixed Signal GUI

**Signal Generation & Classification Dashboard**

This repository contains a PySide6-based graphical application for generating digital communication waveforms, applying channel models and noise, training machine learning models and performing inference. The core functionality is split between a user interface (`main_window.py` and various tabs) and a backend that handles waveform generation (via the MATLAB engine or Python implementations), dataset management, model training (TensorFlow & PyTorch), and inference.

---

## Features

- **Waveform Selection** – Choose modulation types (FSK, PAM, FHSS, etc.) and parameters. The generation routines can call MATLAB functions when the MATLAB Engine API is available.
- **Channel & Noise** – Apply channel models and additive noise to generated signals.
- **ML Training** – Train classification models using TensorFlow or PyTorch on synthetic waveform datasets.
- **Inference Results** – Visualize predictions, confusion matrices and performance metrics.
- **Dataset Management** – Save/load examples in `.npy`/`.json` pairs using a shared `DatasetManager`.

---

## Project Structure
```
main_window.py             # entry point for the PySide6 application
requirements.txt           # Python dependencies
backend/                   # signal logic, dataset & model utilities
    ├── augmentation.py
    ├── core.py
    ├── dataset_generator.py
    ├── dataset_manager.py
    ├── generators.py
    ├── matlab_engine.py    # wrapper for MATLAB Engine API
    ├── tf_models.py
    ├── torch_models.py
    ├── trainer.py
    ├── waveform_pipeline.py
    ├── waveform_service.py
sionna_widget/             # (likely custom widgets for the Sionna library)
styles/                    # stylesheet definitions
tabs/                      # UI tabs for each major workflow step
waveform_functions/        # MATLAB scripts used by the engine (added to path at runtime)
datasets/                  # generated examples and metadata
models/                    # trained model snapshots (`.pth`, etc.)
```

---

## Installation

1. **Prerequisites**
   - Python 3.10+ (tested with 3.11)
   - [MATLAB](https://www.mathworks.com/products/matlab.html) (optional, for waveform generation)
   - MATLAB Engine for Python ([installation guide](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))

2. **Clone the repository**
   ```bash
   git clone <repo-url> mixedsignal-gui
   cd mixedsignal-gui
   ```

3. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # or venv\Scripts\activate on Windows
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **(Optional) Install the MATLAB engine**
   ```bash
   cd /path/to/matlab/extern/engines/python
   python setup.py install
   ```


---

## Running the Application

1. Activate the environment:
   ```bash
   source venv/bin/activate
   ```
2. Launch the GUI:
   ```bash
   python main_window.py
   ```

On the very first launch the application will display a multi‑page setup wizard
(with the rest of the GUI visible behind it).  The wizard:

* gives a **quick tour** of the four main tabs (waveform, channel, ML, inference)
  with explanatory text and arrows to advance;
* lets the user pick folders for models and datasets and shows the detected GPU
  count, offering CPU/GPU mode selection;
* ends with a short tips & tricks page describing navigation and how to reopen
  the wizard later from the Help menu.

Your answers are stored in Qt settings and the wizard is suppressed on subsequent
starts; delete the settings or use the menu item to see it again.

If MATLAB is available and the engine is installed, the app will attempt to start it and add `waveform_functions` to the MATLAB path, allowing you to call custom MATLAB waveform generators. If not, waveform generation will be disabled and the GUI will still launch, but certain buttons will be inactive.

---

## Development Notes

- Stylesheet customization is located in `styles/stylesheet.py`.
- New waveform-generation routines can be added under `backend/generators.py` or as MATLAB scripts.
- The dataset manager stores `.npy` and `.json` pairs in `datasets/` by default; you can change this behaviour in `backend/dataset_manager.py`.
- To add new tabs or widgets, look in the `tabs/` directory and follow the existing patterns.

---

## License

TODO

---

## Acknowledgements

This project makes use of:
- [PySide6](https://doc.qt.io/qtforpython/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Sionna](https://sionna.readthedocs.io/)