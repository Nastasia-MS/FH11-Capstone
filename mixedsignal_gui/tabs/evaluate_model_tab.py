from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QFrame, QFileDialog,
                               QDoubleSpinBox, QSpinBox, QScrollArea, QMessageBox,
                               QTabWidget, QSlider)
from PySide6.QtCore import Qt, QSettings
import os
import json
import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from mixedsignal_gui.widgets.waveform_plots import PlottingWidget, FreqDomainPlot, IQDomainPlot, SpectrogramPlot, _theme_toolbar
from mixedsignal_gui.widgets.toggle_switch import ToggleSwitch
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker
from mixedsignal_gui.widgets.modulation_utils import (mark_unavailable_modulations,
                                                     selected_modulation)
from mixedsignal_gui.backend.augmentation import (AugmentationPipeline, AWGNAugmentation,
                                  ScalarAmplitudeAndPhaseShift, FrequencyShift)

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

        # Waveform defaults (match Waveform Selection tab)
        self.fc = 1e6       # Hz
        self.fs = 8e6       # Hz
        self.var = 1.0
        self.alpha = 0.35
        self.Tsymb = 1e-6   # seconds
        self.M = 4
        self.Nsymb = 256
        self.span = 10

        self.output_type = "passband"

        self._last_signal = None
        self._last_modulation = None

        self.setup_ui()
        install_wheel_blocker(self)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.page_tabs = QTabWidget()

        # ── Page 1: Generate & Classify ──────────────────────────────────
        gen_page = QWidget()
        gen_layout = QHBoxLayout(gen_page)
        gen_layout.setContentsMargins(0, 0, 0, 0)
        gen_layout.setSpacing(10)
        gen_layout.addWidget(self._create_config_panel(), 1)
        gen_layout.addWidget(self._create_gen_results_panel(), 2)
        self.page_tabs.addTab(gen_page, "Generate && Classify")

        # ── Page 2: Channel Test ─────────────────────────────────────────
        chan_page = QWidget()
        chan_layout = QHBoxLayout(chan_page)
        chan_layout.setContentsMargins(0, 0, 0, 0)
        chan_layout.setSpacing(10)
        chan_layout.addWidget(self._create_channel_controls_panel(), 1)
        chan_layout.addWidget(self._create_channel_results_panel(), 2)
        self.page_tabs.addTab(chan_page, "Channel Test")

        root.addWidget(self.page_tabs)

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
                                          "WiFi", "LTE", "5G_NR", "Zigbee"])
        mark_unavailable_modulations(self.waveform_combo, self.eng)
        self.waveform_combo.currentTextChanged.connect(self._on_waveform_changed)
        layout.addWidget(self.waveform_combo)

        # fs — display in MHz, store in Hz (matches Waveform Selection tab)
        layout.addWidget(QLabel("Sampling Frequency fs (MHz)"))
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.1, 1000)
        self.fs_spin.setDecimals(2)
        self.fs_spin.setValue(self.fs / 1e6)
        self.fs_spin.valueChanged.connect(lambda v: setattr(self, "fs", v * 1e6))
        layout.addWidget(self.fs_spin)

        # fc — display in MHz, store in Hz
        layout.addWidget(QLabel("Carrier Frequency fc (MHz)"))
        self.fc_spin = QDoubleSpinBox()
        self.fc_spin.setRange(0.1, 200)
        self.fc_spin.setValue(self.fc / 1e6)
        self.fc_spin.valueChanged.connect(lambda v: setattr(self, "fc", v * 1e6))
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

        # Tsymb — display in µs, store in seconds (matches Waveform Selection tab)
        layout.addWidget(QLabel("Symbol Period Tsymb (µs)"))
        self.tsymb_spin = QDoubleSpinBox()
        self.tsymb_spin.setRange(0.01, 100.0)
        self.tsymb_spin.setValue(self.Tsymb * 1e6)
        self.tsymb_spin.valueChanged.connect(lambda v: setattr(self, "Tsymb", v * 1e-6))
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

        # Output Type (matches Waveform Selection tab)
        layout.addWidget(QLabel("Output Type"))
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(["Passband (Real)", "Baseband (Complex IQ)"])
        self.output_type_combo.currentIndexChanged.connect(
            lambda idx: setattr(self, "output_type", "passband" if idx == 0 else "baseband")
        )
        layout.addWidget(self.output_type_combo)

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

    # ── Generate & Classify — right panel ───────────────────────────────

    def _create_gen_results_panel(self):
        """Right side of the Generate & Classify page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Classification result card
        result_card = QFrame()
        result_card.setObjectName("card")
        rc = QVBoxLayout(result_card)
        rc.setContentsMargins(24, 16, 24, 16)

        self.result_title = QLabel("Classification Result")
        self.result_title.setProperty("class", "section-title")
        rc.addWidget(self.result_title)

        self.result_label = QLabel("—")
        self.result_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #818cf8;")
        rc.addWidget(self.result_label)

        # Probability bar chart
        self.prob_figure = Figure(figsize=(6, 1.8), dpi=100)
        self.prob_canvas = FigureCanvas(self.prob_figure)
        self.prob_canvas.setStyleSheet("background: transparent;")
        self.prob_canvas.setMaximumHeight(160)
        self.prob_toolbar = NavigationToolbar(self.prob_canvas, result_card)
        _theme_toolbar(self.prob_toolbar)
        rc.addWidget(self.prob_toolbar)
        rc.addWidget(self.prob_canvas)

        layout.addWidget(result_card)

        # Waveform plots (same tabs as Waveform Selection)
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

    # ── Channel Test — left panel (controls) ─────────────────────────────

    def _create_channel_controls_panel(self):
        """Left side of the Channel Test page — impairment controls."""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        title = QLabel("Channel Impairments")
        title.setProperty("class", "section-title")
        layout.addWidget(title)
        subtitle = QLabel("Add noise / impairments to the generated waveform and re-classify")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(subtitle)

        # --- AWGN ---
        awgn_header = QHBoxLayout()
        awgn_left = QVBoxLayout()
        awgn_t = QLabel("AWGN (Additive White Gaussian Noise)")
        awgn_t.setProperty("class", "section-title")
        awgn_left.addWidget(awgn_t)
        awgn_header.addLayout(awgn_left)
        awgn_header.addStretch()
        self.ch_awgn_toggle = ToggleSwitch()
        self.ch_awgn_toggle.setChecked(True)
        awgn_header.addWidget(self.ch_awgn_toggle)
        layout.addLayout(awgn_header)

        self.ch_awgn_controls = QWidget()
        awgn_ctrl = QVBoxLayout(self.ch_awgn_controls)
        awgn_ctrl.setContentsMargins(0, 0, 0, 0)
        self.ch_snr_db = 20.0
        awgn_ctrl.addLayout(self._create_slider_control(
            "SNR (dB)", 20.0, "dB", -10, 40, "ch_snr_db"))
        layout.addWidget(self.ch_awgn_controls)

        def _on_ch_awgn_toggled(checked):
            self.ch_awgn_controls.setEnabled(checked)
        self.ch_awgn_toggle.toggled.connect(_on_ch_awgn_toggled)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Amplitude & Phase Shift ---
        ap_header = QHBoxLayout()
        ap_left = QVBoxLayout()
        ap_t = QLabel("Amplitude & Phase Shift")
        ap_t.setProperty("class", "section-title")
        ap_left.addWidget(ap_t)
        ap_header.addLayout(ap_left)
        ap_header.addStretch()
        self.ch_amp_phase_toggle = ToggleSwitch()
        self.ch_amp_phase_toggle.setChecked(False)
        ap_header.addWidget(self.ch_amp_phase_toggle)
        layout.addLayout(ap_header)

        self.ch_amp_phase_controls = QWidget()
        ap_ctrl = QVBoxLayout(self.ch_amp_phase_controls)
        ap_ctrl.setContentsMargins(0, 0, 0, 0)

        self.ch_amplitude = 1.0
        ap_ctrl.addWidget(QLabel("Amplitude Scaling"))
        self.ch_amplitude_spin = QDoubleSpinBox()
        self.ch_amplitude_spin.setRange(0.0, 5.0)
        self.ch_amplitude_spin.setSingleStep(0.1)
        self.ch_amplitude_spin.setValue(1.0)
        self.ch_amplitude_spin.valueChanged.connect(lambda v: setattr(self, 'ch_amplitude', v))
        ap_ctrl.addWidget(self.ch_amplitude_spin)

        self.ch_phase_deg = 0.0
        ap_ctrl.addWidget(QLabel("Phase Shift (degrees)"))
        self.ch_phase_spin = QDoubleSpinBox()
        self.ch_phase_spin.setRange(-180.0, 180.0)
        self.ch_phase_spin.setSingleStep(5.0)
        self.ch_phase_spin.setValue(0.0)
        self.ch_phase_spin.valueChanged.connect(lambda v: setattr(self, 'ch_phase_deg', v))
        ap_ctrl.addWidget(self.ch_phase_spin)

        self.ch_amp_phase_controls.setEnabled(False)
        layout.addWidget(self.ch_amp_phase_controls)

        def _on_ch_ap_toggled(checked):
            self.ch_amp_phase_controls.setEnabled(checked)
        self.ch_amp_phase_toggle.toggled.connect(_on_ch_ap_toggled)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # --- Frequency Offset ---
        fo_header = QHBoxLayout()
        fo_left = QVBoxLayout()
        fo_t = QLabel("Frequency Offset (CFO)")
        fo_t.setProperty("class", "section-title")
        fo_left.addWidget(fo_t)
        fo_header.addLayout(fo_left)
        fo_header.addStretch()
        self.ch_freq_shift_toggle = ToggleSwitch()
        self.ch_freq_shift_toggle.setChecked(False)
        fo_header.addWidget(self.ch_freq_shift_toggle)
        layout.addLayout(fo_header)

        self.ch_freq_shift_controls = QWidget()
        fs_ctrl = QVBoxLayout(self.ch_freq_shift_controls)
        fs_ctrl.setContentsMargins(0, 0, 0, 0)

        self.ch_freq_shift_hz = 0.0
        fs_ctrl.addWidget(QLabel("Frequency Offset (MHz)"))
        self.ch_freq_shift_spin = QDoubleSpinBox()
        self.ch_freq_shift_spin.setRange(-100.0, 100.0)
        self.ch_freq_shift_spin.setSingleStep(0.01)
        self.ch_freq_shift_spin.setDecimals(4)
        self.ch_freq_shift_spin.setValue(0.0)
        self.ch_freq_shift_spin.valueChanged.connect(
            lambda v: setattr(self, 'ch_freq_shift_hz', v * 1e6))
        fs_ctrl.addWidget(self.ch_freq_shift_spin)

        self.ch_freq_shift_controls.setEnabled(False)
        layout.addWidget(self.ch_freq_shift_controls)

        def _on_ch_fs_toggled(checked):
            self.ch_freq_shift_controls.setEnabled(checked)
        self.ch_freq_shift_toggle.toggled.connect(_on_ch_fs_toggled)

        # --- Apply & Classify ---
        layout.addSpacing(8)
        self.ch_apply_btn = QPushButton("▶ Apply Impairments & Classify")
        self.ch_apply_btn.setObjectName("primaryButton")
        self.ch_apply_btn.setMinimumHeight(36)
        self.ch_apply_btn.clicked.connect(self._channel_test_classify)
        layout.addWidget(self.ch_apply_btn)

        self.ch_status_label = QLabel("Generate a waveform first, then apply impairments.")
        self.ch_status_label.setProperty("class", "stat-label")
        self.ch_status_label.setWordWrap(True)
        layout.addWidget(self.ch_status_label)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("card")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    # ── Channel Test — right panel (results + comparison) ────────────────

    def _create_channel_results_panel(self):
        """Right side of the Channel Test page — classification result + comparison."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Compact classification result row
        result_card = QFrame()
        result_card.setObjectName("card")
        rc = QHBoxLayout(result_card)
        rc.setContentsMargins(16, 8, 16, 8)

        ch_title = QLabel("Result:")
        ch_title.setProperty("class", "section-title")
        rc.addWidget(ch_title)

        self.ch_result_label = QLabel("—")
        self.ch_result_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #818cf8;")
        rc.addWidget(self.ch_result_label)
        rc.addStretch()

        # Probability bar chart (compact)
        self.ch_prob_figure = Figure(figsize=(4, 1.2), dpi=100)
        self.ch_prob_canvas = FigureCanvas(self.ch_prob_figure)
        self.ch_prob_canvas.setStyleSheet("background: transparent;")
        self.ch_prob_canvas.setFixedHeight(100)
        self.ch_prob_canvas.setMinimumWidth(300)
        rc.addWidget(self.ch_prob_canvas)

        # Toolbar hidden — use save from comparison widget instead
        self.ch_prob_toolbar = None

        layout.addWidget(result_card)

        # Comparison widget (clean vs augmented) — gets all remaining space
        from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget
        self.ch_comparison_plot = ComparisonWidget()
        layout.addWidget(self.ch_comparison_plot, 1)
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

    # Presets store internal units (Hz, seconds) — spin boxes display MHz / µs
    _WAVEFORM_PRESETS = {
        'PAM':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 4},
        'QAM':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 16},
        'PSK':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 8},
        'FSK':    {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 4},
        'FHSS':   {'fs': 48e3, 'fc': 6e3, 'Tsymb': 1e-3, 'M': 8},
        'LFM':    {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4, 'M': 4},
        'Barker': {'fs': 1e6, 'fc': 200e3, 'Tsymb': 5e-5, 'M': 7},
        'FMCW':   {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4, 'M': 4},
        # Standards waveforms are generated at their native 30.72 Msps so the
        # occupied bandwidth matches what the Waveform tab produces; evaluating
        # a model at 1 MHz against 30.72 Msps training data would resample the
        # signal into a different fraction of the band.
        'WiFi':   {'fs': 30.72e6, 'fc': 10e6, 'M': 4},
        'LTE':    {'fs': 30.72e6, 'fc': 10e6, 'M': 4},
        '5G_NR':  {'fs': 30.72e6, 'fc': 10e6, 'M': 4},
        'Zigbee': {'fs': 30.72e6, 'fc': 10e6, 'M': 4},
    }

    def _on_waveform_changed(self, modulation):
        """Update parameter spin-boxes with sensible defaults for the selected waveform."""
        # currentTextChanged delivers the displayed label, which may carry the
        # "(needs MATLAB)" annotation; presets are keyed on the bare name.
        modulation = modulation.split()[0] if modulation else modulation
        preset = self._WAVEFORM_PRESETS.get(modulation, {})
        if not preset:
            return
        # Convert to display units (MHz, µs) before setting spin-box values
        if 'fs' in preset:
            self.fs_spin.setValue(preset['fs'] / 1e6)
        if 'fc' in preset:
            self.fc_spin.setValue(preset['fc'] / 1e6)
        if 'Tsymb' in preset:
            self.tsymb_spin.setValue(preset['Tsymb'] * 1e6)
        if 'M' in preset:
            self.M_spin.setValue(preset['M'])

    # ------------------------------------------------------------------
    # Generate waveform + classify
    # ------------------------------------------------------------------

    def generate_and_classify(self):
        modulation = selected_modulation(self.waveform_combo)
        fs = float(self.fs)
        tsymb = float(self.Tsymb)
        fc = float(self.fc)
        m = float(self.M)
        var = float(self.var)
        nsymb = int(self.Nsymb)
        alpha = float(self.alpha)
        span = int(self.span)
        pulse_shape = self.pulse_shape_combo.currentText()

        # Enforce Nyquist (matches Waveform Selection tab)
        if fc >= fs / 2:
            QMessageBox.warning(
                self, "Invalid Parameters",
                f"fc={fc:.2e} Hz must be < fs/2={fs/2:.2e} Hz")
            return

        # Validate: fs * Tsymb must be an integer (samples per symbol)
        sps_raw = fs * tsymb
        if abs(sps_raw - round(sps_raw)) > 1e-9:
            QMessageBox.warning(
                self, "Invalid Parameters",
                f"fs × Tsymb = {sps_raw:.6f} must be an integer (samples per symbol)")
            return

        # --- Generate waveform via WaveformPipeline ---
        try:
            from mixedsignal_gui.backend.waveform_pipeline import WaveformPipeline
            pipeline = WaveformPipeline(self.eng)
            result = pipeline.generate(
                fs=fs, Tsymb=tsymb, Nsymb=nsymb, fc=fc, M=m,
                modulation=modulation, var=var,
                alpha=alpha, span=span, pulse_shape=pulse_shape,
                output_type=self.output_type,
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
        baseband_symbols = result.get("baseband_symbols")

        # --- Update plots (matches Waveform Selection tab) ---
        t = np.arange(len(data)) / fs * 1e6   # time axis in µs

        # Time domain: for complex data plot I and Q; for real plot as-is
        if np.iscomplexobj(data):
            self.waveform_plot.plot_data(t, np.real(data), np.imag(data))
        else:
            self.waveform_plot.plot_data(t, np.real(data))

        # Frequency domain: for complex use fftshift for centered spectrum
        if np.iscomplexobj(data):
            ft = np.fft.fftshift(np.fft.fft(data))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(data), 1 / fs)) * 1e-6
        else:
            ft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1 / fs) * 1e-6

        self.freq_plot.plot_data(freqs, np.abs(ft))

        self.constellation_plot.plot_data(
            data=data, fs=fs, fc=fc, sps=sps, M=m,
            modulation=modulation, nsymb=nsymb,
            baseband_symbols=baseband_symbols,
        )
        self.spectrogram_plot.plot_data(x=data, fs=fs, modulation=modulation)

        # --- Classify ---
        if self.model is None:
            self.result_label.setText("No model loaded")
            self.status_label.setText("Load a model to classify waveforms.")
            return

        self._classify_signal(data)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_signal(self, data, target='generate'):
        """Preprocess one signal and run inference.

        target: 'generate' updates the Generate & Classify tab,
                'channel' updates the Channel Test tab.
        """
        from mixedsignal_gui.backend.trainer import TrainerThread
        target_len = TrainerThread.TARGET_LENGTH

        # Keep the native dtype: casting to float32 here used to discard Q
        # before the IQ branch could ever see it.
        raw = np.asarray(data).ravel()
        if len(raw) > target_len:
            raw = raw[:target_len]
        elif len(raw) < target_len:
            raw = np.pad(raw, (0, target_len - len(raw)))

        # Match the channel count the model was actually trained with.  Older
        # models predate that metadata, so fall back to the architecture name.
        meta = self.model_metadata or {}
        expected_ch = int(meta.get(
            'input_channels', 2 if meta.get('model_name', '') in IQ_MODELS else 1))

        if expected_ch == 2:
            if not np.iscomplexobj(raw):
                msg = ("This model expects 2-channel I/Q, but the signal is real "
                       "(passband). Set Output Type to Baseband (Complex IQ) and "
                       "regenerate.")
                if target == 'channel':
                    self.ch_status_label.setText(msg)
                else:
                    self.status_label.setText(msg)
                return
            X = np.stack([raw.real, raw.imag], axis=0)[np.newaxis].astype(np.float32)
            X = self._normalize_iq(X)
        else:
            # A real model on complex input: use I, which is what a real
            # receiver front-end would deliver.
            X = np.real(raw).astype(np.float32)[np.newaxis, np.newaxis, :]

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

        if target == 'channel':
            self.ch_result_label.setText(f"{pred_name}  ({confidence:.1f}%)")
            self.ch_status_label.setText(
                f"Augmented {self._last_modulation} → classified as {pred_name}")
            self._plot_probabilities(probs, target='channel')
        else:
            self.result_label.setText(f"{pred_name}  ({confidence:.1f}%)")
            self.status_label.setText(
                f"Generated {self._last_modulation} waveform → classified as {pred_name}"
            )
            self._plot_probabilities(probs, target='generate')

    def _plot_probabilities(self, probs, target='generate'):
        """Draw horizontal bar chart of class probabilities."""
        if target == 'channel':
            fig, canvas = self.ch_prob_figure, self.ch_prob_canvas
        else:
            fig, canvas = self.prob_figure, self.prob_canvas

        fig.clear()
        ax = fig.add_subplot(111)

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

        _apply_theme_style(fig, self)
        fig.tight_layout()
        canvas.draw()

    # ------------------------------------------------------------------
    # Slider helper (matches ChannelNoiseTab)
    # ------------------------------------------------------------------

    def _create_slider_control(self, label, value, unit, min_val, max_val, attr_name):
        """Create a slider control with label and value display."""
        container = QVBoxLayout()
        container.setSpacing(4)

        header = QHBoxLayout()
        label_widget = QLabel(label)
        value_label = QLabel(f"{int(value)} {unit}")
        value_label.setProperty("class", "stat-value")
        header.addWidget(label_widget)
        header.addStretch()
        header.addWidget(value_label)
        container.addLayout(header)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(value))

        def update_value(v):
            value_label.setText(f"{v} {unit}")
            setattr(self, attr_name, float(v))

        slider.valueChanged.connect(update_value)
        container.addWidget(slider)

        return container

    # ------------------------------------------------------------------
    # Channel Test — apply impairments and re-classify
    # ------------------------------------------------------------------

    def _channel_test_classify(self):
        """Apply channel impairments to the last generated signal and classify."""
        if self._last_signal is None:
            QMessageBox.warning(
                self, "No Signal",
                "Generate a waveform first in the 'Generate & Classify' tab.")
            return

        clean = self._last_signal
        fs = float(self.fs)

        # Build augmentation pipeline (same as ChannelNoiseTab AWGN path)
        pipeline = AugmentationPipeline()
        if self.ch_awgn_toggle.isChecked():
            pipeline.add(AWGNAugmentation(snr_db=self.ch_snr_db))
        if self.ch_amp_phase_toggle.isChecked():
            phase_rad = np.deg2rad(self.ch_phase_deg)
            pipeline.add(ScalarAmplitudeAndPhaseShift(
                amplitude=self.ch_amplitude, phi=phase_rad))
        if self.ch_freq_shift_toggle.isChecked():
            pipeline.add(FrequencyShift(delta_f=self.ch_freq_shift_hz))

        augmented = pipeline.apply(clean, fs)

        # Update comparison plot
        modulation = self._last_modulation
        fc = float(self.fc)
        tsymb = float(self.Tsymb)
        sps = int(round(fs * tsymb))
        m = float(self.M)
        nsymb = int(self.Nsymb)
        alpha = float(self.alpha)
        span = int(self.span)
        pulse_shape = self.pulse_shape_combo.currentText()

        self.ch_comparison_plot.plot_comparison(
            clean_signal=clean,
            augmented_signal=augmented,
            fs=fs,
            fc=fc,
            sps=sps,
            modulation=modulation,
            M=m,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            nsymb=nsymb,
        )

        # Classify the augmented signal
        if self.model is None:
            self.ch_result_label.setText("No model loaded")
            self.ch_status_label.setText("Load a model to classify.")
            return

        self.ch_status_label.setText(
            f"Applied impairments to {self._last_modulation} waveform — classifying…")
        self._classify_signal(augmented, target='channel')

    # ------------------------------------------------------------------
    # IQ helpers (same as trainer / inference)
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_iq(X_flat):
        """Split complex baseband (N, L) into (N, 2, L) I/Q channels.

        Complex only — the old even/odd fallback silently produced nonsense
        for real passband signals, whose consecutive samples are not I and Q.
        """
        if not np.iscomplexobj(X_flat):
            raise ValueError("_prepare_iq expects complex baseband input.")
        return np.stack([X_flat.real, X_flat.imag], axis=1).astype(np.float32)

    @staticmethod
    def _normalize_iq(X_iq):
        power = np.mean(X_iq[:, 0, :] ** 2 + X_iq[:, 1, :] ** 2, axis=1, keepdims=True)
        power = np.maximum(power, 1e-10)
        scale = np.sqrt(power)[:, np.newaxis, :]
        return X_iq / scale