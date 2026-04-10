from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QFrame, QFileDialog,
                               QDoubleSpinBox, QSpinBox, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt, QSettings
import os
import json
import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from mixedsignal_gui.widgets.waveform_plots import PlottingWidget, FreqDomainPlot, IQDomainPlot, SpectrogramPlot
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker

# Must stay in sync with trainer.py
IQ_MODELS = {'ResNet1DOptimized'}


def _apply_theme_style(fig, widget):
    """Apply the widget's palette colours to a matplotlib figure so it matches the app theme."""
    palette = widget.palette()
    bg   = palette.window().color().name()
    fg   = palette.windowText().color().name()
    axes_bg = palette.base().color().name()
    fig.patch.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(axes_bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)


class EvaluateModelTab(QWidget):
    """Generate a waveform and classify it with a loaded model."""

    def __init__(self, eng, dataset_manager=None, parent=None):
        super().__init__(parent)
        self.eng = eng
        self.dataset_manager = dataset_manager

        # Model state
        self.model = None
        self.model_path = None
        self.model_metadata = None
        self.class_labels = []

        # Waveform defaults
        self.fs = 48000
        self.Tsymb = 0.001
        self.fc = 6000
        self.M = 16
        self.var = 1.0
        self.Nsymb = 2048
        self.alpha = 0.35
        self.span = 8

        self._last_signal = None
        self._last_modulation = None

        self.setup_ui()
        install_wheel_blocker(self)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Left: config panel
        left = self._create_config_panel()
        layout.addWidget(left, 1)

        # Right: plots + classification result
        right = self._create_results_panel()
        layout.addWidget(right, 2)

    def _create_config_panel(self):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        # --- Model section ---
        title = QLabel("Evaluate Model")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        subtitle = QLabel("Generate a waveform and classify it")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(subtitle)

        model_lbl = QLabel("Model")
        model_lbl.setProperty("class", "stat-label")
        layout.addWidget(model_lbl)

        model_btn_row = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self.load_model)
        model_btn_row.addWidget(self.load_model_btn)
        layout.addLayout(model_btn_row)

        self.model_label = QLabel("No model loaded")
        self.model_label.setProperty("class", "stat-label")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        # --- Waveform params (mirrors WaveformSelectionTab) ---
        layout.addSpacing(8)
        wf_lbl = QLabel("Waveform Parameters")
        wf_lbl.setProperty("class", "stat-label")
        layout.addWidget(wf_lbl)

        layout.addWidget(QLabel("Waveform"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(["PAM", "QAM", "PSK", "FSK", "FHSS",
                                          "LFM", "Barker", "FMCW",
                                          "WiFi", "LTE", "5G_NR"])
        self.waveform_combo.currentTextChanged.connect(self._on_waveform_changed)
        layout.addWidget(self.waveform_combo)

        layout.addWidget(QLabel("Sampling Frequency fs (Hz)"))
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1, 1e9)
        self.fs_spin.setDecimals(0)
        self.fs_spin.setSingleStep(1000)
        self.fs_spin.setValue(self.fs)
        self.fs_spin.valueChanged.connect(lambda v: setattr(self, "fs", v))
        layout.addWidget(self.fs_spin)

        layout.addWidget(QLabel("Carrier Frequency fc (Hz)"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0, 1e9)
        self.fc_spin.setDecimals(0)
        self.fc_spin.setSingleStep(1000)
        self.fc_spin.setValue(self.fc)
        self.fc_spin.valueChanged.connect(lambda v: setattr(self, "fc", v))
        layout.addWidget(self.fc_spin)

        layout.addWidget(QLabel("Noise Variance"))
        self.var_spin = QDoubleSpinBox()
        self.var_spin.setRange(0.0, 10.0)
        self.var_spin.setSingleStep(0.1)
        self.var_spin.setValue(self.var)
        self.var_spin.valueChanged.connect(lambda v: setattr(self, "var", v))
        layout.addWidget(self.var_spin)

        layout.addWidget(QLabel("RRC Roll-off α"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.valueChanged.connect(lambda v: setattr(self, "alpha", v))
        layout.addWidget(self.alpha_spin)

        layout.addWidget(QLabel("Symbol Period Tsymb (s)"))
        self.tsymb_spin = QDoubleSpinBox()
        self.tsymb_spin.setRange(1e-7, 1.0)
        self.tsymb_spin.setDecimals(6)
        self.tsymb_spin.setSingleStep(0.0001)
        self.tsymb_spin.setValue(self.Tsymb)
        self.tsymb_spin.valueChanged.connect(lambda v: setattr(self, "Tsymb", v))
        layout.addWidget(self.tsymb_spin)

        layout.addWidget(QLabel("Modulation Order M"))
        self.M_spin = QDoubleSpinBox()
        self.M_spin.setRange(2, 256)
        self.M_spin.setDecimals(0)
        self.M_spin.setSingleStep(1)
        self.M_spin.setValue(self.M)
        self.M_spin.valueChanged.connect(lambda v: setattr(self, "M", v))
        layout.addWidget(self.M_spin)

        layout.addWidget(QLabel("Number of Symbols"))
        self.nsymb_spin = QDoubleSpinBox()
        self.nsymb_spin.setRange(16, 10000)
        self.nsymb_spin.setDecimals(0)
        self.nsymb_spin.setSingleStep(1)
        self.nsymb_spin.setValue(self.Nsymb)
        self.nsymb_spin.valueChanged.connect(lambda v: setattr(self, "Nsymb", v))
        layout.addWidget(self.nsymb_spin)

        layout.addWidget(QLabel("Pulse Span (symbols)"))
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setRange(2, 50)
        self.span_spin.setDecimals(0)
        self.span_spin.setSingleStep(1)
        self.span_spin.setValue(self.span)
        self.span_spin.valueChanged.connect(lambda v: setattr(self, "span", v))
        layout.addWidget(self.span_spin)

        layout.addWidget(QLabel("Pulse Shape"))
        self.pulse_shape_combo = QComboBox()
        self.pulse_shape_combo.addItems(["rrc", "rect"])
        layout.addWidget(self.pulse_shape_combo)

        # --- Action buttons ---
        layout.addSpacing(8)
        self.generate_btn = QPushButton("▶ Generate & Classify")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self.generate_and_classify)
        layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Load a model, then generate a waveform.")
        self.status_label.setProperty("class", "stat-label")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("card")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    def _create_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Classification result card
        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(24, 16, 24, 16)

        self.result_title = QLabel("Classification Result")
        self.result_title.setProperty("class", "section-title")
        result_layout.addWidget(self.result_title)

        self.result_label = QLabel("—")
        self.result_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #818cf8;")
        result_layout.addWidget(self.result_label)

        # Probability bar chart
        self.prob_figure = Figure(figsize=(6, 1.8), dpi=100)
        self.prob_canvas = FigureCanvas(self.prob_figure)
        self.prob_canvas.setMaximumHeight(160)
        result_layout.addWidget(self.prob_canvas)

        layout.addWidget(result_card)

        # Waveform plots (same tabs as Waveform Selection)
        from PySide6.QtWidgets import QTabWidget
        self.plot_tabs = QTabWidget()
        self.waveform_plot = PlottingWidget()
        self.freq_plot = FreqDomainPlot()
        self.constellation_plot = IQDomainPlot()
        self.spectrogram_plot = SpectrogramPlot()

        self.plot_tabs.addTab(self.waveform_plot, "Waveform")
        self.plot_tabs.addTab(self.freq_plot, "Frequency")
        self.plot_tabs.addTab(self.constellation_plot, "Constellation")
        self.plot_tabs.addTab(self.spectrogram_plot, "Spectrogram")

        layout.addWidget(self.plot_tabs, 1)
        return panel

    # ------------------------------------------------------------------
    # Model loading (same as InferenceResultsTab)
    # ------------------------------------------------------------------
    
    def get_device(self):
        """Get the current device from settings (read fresh each time)."""
        settings = QSettings("MyCompany", "MixedSignalGUI")
        compute_mode = settings.value("mode", "CPU")
        if compute_mode == "GPU" and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _load_metadata(self, pth_path):
        meta_path = pth_path.rsplit('.', 1)[0] + '.json'
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: could not read metadata {meta_path}: {e}")
        return None

    def load_model(self):
        # Get default model directory from settings
        settings = QSettings("MyCompany", "MixedSignalGUI")
        default_dir = settings.value("modelPath", "")
        
        # If default not set, try to find models folder
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = os.path.join(os.getcwd(), 'models')
            if not os.path.isdir(default_dir):
                default_dir = ""
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", default_dir,
            "PyTorch Models (*.pth);;All Files (*)"
        )
        if not filepath:
            return
        self._do_load_model(filepath)

    def receive_trained_model(self, model_path, labels):
        """Slot: auto-load a freshly trained model."""
        self.class_labels = list(labels) if labels else []
        self._do_load_model(model_path)

    def _do_load_model(self, filepath):
        try:
            from mixedsignal_gui.backend.torch_models import get_model

            meta = self._load_metadata(filepath)
            self.model_metadata = meta

            if meta:
                model_name = meta.get('model_name', 'SimpleCNN')
                num_classes = meta.get('num_classes', 2)
                signal_length = meta.get('signal_length', 256)
                in_channels = meta.get('input_channels', 1)
                if meta.get('class_labels'):
                    self.class_labels = meta['class_labels']
            else:
                model_name = 'SimpleCNN'
                num_classes = 2
                signal_length = 256
                in_channels = 1

            # Get current device setting
            device = self.get_device()
            
            # Load model to the correct device
            state_dict = torch.load(filepath, map_location=device, weights_only=True)
            model = get_model(
                model_name,
                num_classes=num_classes,
                input_size=signal_length,
                in_channels=in_channels,
            )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self.model = model
            self.model_path = filepath

            device_name = "GPU" if device == 'cuda' else "CPU"
            info = f"Loaded: {os.path.basename(filepath)} ({model_name}, {num_classes} classes) on {device_name}"
            self.model_label.setText(info)
            self.status_label.setText("Model ready. Generate a waveform to classify.")
        except Exception as e:
            self.model_label.setText(f"Failed: {e}")
            print(f"Error loading model: {e}")

    # ------------------------------------------------------------------
    # Smart defaults per waveform type
    # ------------------------------------------------------------------

    _WAVEFORM_PRESETS = {
        'PAM':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 4},
        'QAM':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 16},
        'PSK':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 8},
        'FSK':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 4},
        'FHSS':   {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 8},
        'LFM':    {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4, 'M': 4},
        'Barker': {'fs': 1e6, 'fc': 200e3, 'Tsymb': 5e-5, 'M': 7},
        'FMCW':   {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4, 'M': 4},
        'WiFi':   {'fs': 1e6, 'fc': 250e3, 'M': 4},
        'LTE':    {'fs': 1e6, 'fc': 250e3, 'M': 4},
        '5G_NR':  {'fs': 1e6, 'fc': 250e3, 'M': 4},
    }

    def _on_waveform_changed(self, modulation):
        """Update parameter spin-boxes with sensible defaults for the selected waveform."""
        preset = self._WAVEFORM_PRESETS.get(modulation, {})
        if not preset:
            return
        if 'fs' in preset:
            self.fs_spin.setValue(preset['fs'])
        if 'fc' in preset:
            self.fc_spin.setValue(preset['fc'])
        if 'Tsymb' in preset:
            self.tsymb_spin.setValue(preset['Tsymb'])
        if 'M' in preset:
            self.M_spin.setValue(preset['M'])

    # ------------------------------------------------------------------
    # Generate waveform + classify
    # ------------------------------------------------------------------

    def generate_and_classify(self):
        modulation = self.waveform_combo.currentText()
        fs = float(self.fs)
        tsymb = float(self.Tsymb)
        fc = float(self.fc)
        m = float(self.M)
        var = float(self.var)
        nsymb = int(self.Nsymb)
        alpha = float(self.alpha)
        span = int(self.span)
        pulse_shape = self.pulse_shape_combo.currentText()

        # --- Generate waveform via WaveformPipeline ---
        try:
            from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
            pipeline = WaveformPipeline(self.eng)
            result = pipeline.generate(
                fs=fs, Tsymb=tsymb, Nsymb=nsymb, fc=fc, M=m,
                modulation=modulation, var=var,
                alpha=alpha, span=span, pulse_shape=pulse_shape,
                output_type='passband',
            )
            data = result['signal']
            sps = int(round(fs * tsymb))
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Parameters", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Waveform generation failed: {e}")
            return

        self._last_signal = data
        self._last_modulation = modulation

        # --- Update plots ---
        T = len(data) / fs
        t = np.linspace(0, T, len(data))

        # Compute frequency domain in Python (no MATLAB call needed)
        if np.iscomplexobj(data):
            ft = np.fft.fftshift(np.fft.fft(data))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1 / fs))
        else:
            ft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1 / fs)

        self.waveform_plot.plot_data(t, data)
        self.freq_plot.plot_data(freqs, np.abs(ft))
        self.spectrogram_plot.plot_data(data, fs, modulation=modulation)
        self.constellation_plot.plot_data(
            data=data, fs=fs, fc=fc, sps=sps, M=m,
            modulation=modulation, alpha=alpha, span=span,
            pulse_shape=pulse_shape, nsymb=nsymb,
        )

        # --- Classify ---
        if self.model is None:
            self.result_label.setText("No model loaded")
            self.status_label.setText("Load a model to classify waveforms.")
            return

        self._classify_signal(data)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_signal(self, data):
        """Preprocess one signal and run inference."""
        from mixedsignal_gui.backend.trainer import TrainerThread
        target_len = TrainerThread.TARGET_LENGTH

        raw = np.asarray(data, dtype=np.float32).ravel()
        if len(raw) > target_len:
            raw = raw[:target_len]
        elif len(raw) < target_len:
            raw = np.pad(raw, (0, target_len - len(raw)))

        is_iq = (self.model_metadata or {}).get('model_name', '') in IQ_MODELS
        if is_iq:
            X = raw[np.newaxis, :]  # (1, L)
            X = self._prepare_iq(X)
            X = self._normalize_iq(X)
        else:
            X = raw[np.newaxis, np.newaxis, :]  # (1, 1, L)

        tensor = torch.from_numpy(X).to(self.get_device())
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        if self.class_labels and pred_idx < len(self.class_labels):
            pred_name = self.class_labels[pred_idx]
        else:
            pred_name = f"Class {pred_idx}"

        confidence = probs[pred_idx] * 100
        self.result_label.setText(f"{pred_name}  ({confidence:.1f}%)")
        self.status_label.setText(
            f"Generated {self._last_modulation} waveform → classified as {pred_name}"
        )

        self._plot_probabilities(probs)

    def _plot_probabilities(self, probs):
        """Draw horizontal bar chart of class probabilities."""
        self.prob_figure.clear()
        ax = self.prob_figure.add_subplot(111)

        n = len(probs)
        labels = self.class_labels[:n] if self.class_labels else [f"C{i}" for i in range(n)]
        y_pos = np.arange(n)

        colors = ['#818cf8' if p < max(probs) else '#4ade80' for p in probs]
        ax.barh(y_pos, probs * 100, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel('Confidence (%)', fontsize=9)
        ax.invert_yaxis()

        _apply_theme_style(self.prob_figure, self)
        self.prob_figure.tight_layout()
        self.prob_canvas.draw()

    # ------------------------------------------------------------------
    # IQ helpers (same as trainer / inference)
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_iq(X_flat):
        if np.iscomplexobj(X_flat):
            return np.stack([X_flat.real, X_flat.imag], axis=1)
        L = X_flat.shape[1]
        if L % 2 != 0:
            X_flat = X_flat[:, :L - 1]
            L -= 1
        return np.stack([X_flat[:, 0::2], X_flat[:, 1::2]], axis=1)

    @staticmethod
    def _normalize_iq(X_iq):
        power = np.mean(X_iq[:, 0, :] ** 2 + X_iq[:, 1, :] ** 2, axis=1, keepdims=True)
        power = np.maximum(power, 1e-10)
        scale = np.sqrt(power)[:, np.newaxis, :]
        return X_iq / scale