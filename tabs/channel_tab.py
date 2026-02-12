from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QSlider, QGridLayout, QComboBox, QDoubleSpinBox)
from PySide6.QtCore import Qt

from widgets.toggle_switch import ToggleSwitch
from widgets.noise_spectrum import NoiseSpectrumWidget
from widgets.waveform_plots import PlottingWidget
from backend.augmentation import AugmentationPipeline, AWGNAugmentation, ScalarAmplitudeAndPhaseShift
import numpy as np


class ChannelNoiseTab(QWidget):
    """Channel configuration and noise settings tab"""
    
    def __init__(self, dataset_manager):
        super().__init__()
        self.dataset_manager = dataset_manager

        # Augmentation params
        self.snr_db = 20.0
        self.amplitude = 1.0
        self.phase_deg = 0.0
        self.awgn_enabled = True
        self.amp_phase_enabled = False

        self.setup_ui()
        self.refresh_dataset_list()

        if self.dataset_manager.get_active():
            self.display_original_signal()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - Channel Configuration
        left_panel = self.create_controls_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel - Noise Configuration and Spectrum
        right_panel = self.create_visualization_panel()
        layout.addWidget(right_panel, 2)
    
    def create_controls_panel(self):
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("📡 Channel Augmentation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Apply channel effects and noise")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Dataset Selection
        dataset_selection = QVBoxLayout()
        dataset_label = QLabel("Select Dataset")
        dataset_label.setProperty("class", "section-title")
        dataset_selection.addWidget(dataset_label)

        # Dataset dropdown
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_selection.addWidget(self.dataset_combo)

        # Upload and Refresh buttons
        dataset_buttons = QHBoxLayout()
        upload_btn = QPushButton("Upload Dataset")
        upload_btn.clicked.connect(self.upload_dataset)
        dataset_buttons.addWidget(upload_btn)

        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.clicked.connect(self.refresh_dataset_list)
        dataset_buttons.addWidget(refresh_btn)
        dataset_selection.addLayout(dataset_buttons)

        layout.addLayout(dataset_selection)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # AWGN Configuration
        awgn_section = QVBoxLayout()

        # AWGN Toggle
        awgn_header = QHBoxLayout()
        awgn_left = QVBoxLayout()
        awgn_title = QLabel("AWGN (Additive White Gaussian Noise)")
        awgn_title.setProperty("class", "section-title")
        awgn_desc = QLabel("Add white Gausian noise")
        awgn_desc.setProperty("class", "section-subtitle")
        awgn_left.addWidget(awgn_title)
        awgn_left.addWidget(awgn_desc)
        awgn_section.addLayout(awgn_left)
        awgn_section.addStretch()
        self.awgn_toggle = ToggleSwitch()
        self.awgn_toggle.setChecked(self.awgn_enabled)
        self.awgn_toggle.toggled.connect(lambda checked: setattr(self, 'awgn_enabled', checked))
        awgn_header.addWidget(self.awgn_toggle)
        awgn_section.addLayout(awgn_header)

        # SNR Slider
        snr_slider_layout = self.create_slider_control(
            "SNR (dB)", self.snr_db, "dB", -10, 40, "snr_db"
        )
        awgn_section.addLayout(snr_slider_layout)

        layout.addLayout(awgn_section)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        # Amplitude & Phase Shift Configuration
        amp_phase_section = QVBoxLayout()

        # Amplitude/Phase Toggle
        amp_phase_header = QHBoxLayout()
        amp_phase_left = QVBoxLayout()
        amp_phase_title = QLabel("Amplitude & Phase Shift")
        amp_phase_title.setProperty("class", "section-title")
        amp_phase_desc = QLabel("Scale amplitude and rotate phase")
        amp_phase_desc.setProperty("class", "section-subtitle")
        amp_phase_left.addWidget(amp_phase_title)
        amp_phase_left.addWidget(amp_phase_desc)
        amp_phase_header.addLayout(amp_phase_left)
        amp_phase_header.addStretch()
        self.amp_phase_toggle = ToggleSwitch()
        self.amp_phase_toggle.setChecked(self.amp_phase_enabled)
        self.amp_phase_toggle.toggled.connect(lambda checked: setattr(self, 'amp_phase_enabled', checked))
        amp_phase_header.addWidget(self.amp_phase_toggle)
        amp_phase_section.addLayout(amp_phase_header)

        # Amplitude Control
        amp_label = QLabel("Amplitude Scaling")
        amp_phase_section.addWidget(amp_label)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 5.0)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setValue(self.amplitude)
        self.amplitude_spin.valueChanged.connect(lambda v: setattr(self, 'amplitude', v))
        amp_phase_section.addWidget(self.amplitude_spin)
        
        # Phase Control
        phase_label = QLabel("Phase Shift (degrees)")
        amp_phase_section.addWidget(phase_label)
        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-180.0, 180.0)
        self.phase_spin.setSingleStep(5.0)
        self.phase_spin.setValue(self.phase_deg)
        self.phase_spin.valueChanged.connect(lambda v: setattr(self, 'phase_deg', v))
        amp_phase_section.addWidget(self.phase_spin)

        layout.addLayout(amp_phase_section)
        
        layout.addStretch()

        # Apply Augmentation Button
        apply_btn = QPushButton("Apply Augmentations")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self.apply_augmentations)
        layout.addWidget(apply_btn)

        # Save Augmented Dataset Button
        save_btn = QPushButton("Save Augmented Dataset")
        save_btn.clicked.connect(self.save_augmented_dataset)
        layout.addWidget(save_btn)
        
        return panel
    
    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Original Signal Card
        comparison_card = QFrame()
        comparison_card.setObjectName("card")
        comparison_layout = QVBoxLayout(comparison_card)
        comparison_layout.setContentsMargins(24, 24, 24, 24)
        
        comparison_title = QLabel("Clean vs Augmented Signal")
        comparison_title.setProperty("class", "card-title")
        comparison_layout.addWidget(comparison_title)
        
        from widgets.comparison_widget import ComparisonWidget
        self.comparison_plot = ComparisonWidget()
        comparison_layout.addWidget(self.comparison_plot)

        layout.addWidget(comparison_card)
        
        return panel
    
    def create_slider_control(self, label, value, unit, min_val, max_val, attr_name):
        """Create a slider control with label and value display"""
        container = QVBoxLayout()
        container.setSpacing(8)
        
        # Label and value
        header = QHBoxLayout()
        label_widget = QLabel(label)
        value_label = QLabel(f"{value} {unit}")
        value_label.setProperty("class", "stat-value")
        value_label.setMinimumHeight(24)
        header.addWidget(label_widget)
        header.addStretch()
        header.addWidget(value_label)
        container.addLayout(header)
        
        container.addSpacing(4)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(value))

        def update_value(v):
            value_label.setText(f"{v} {unit}")
            setattr(self, attr_name, float(v))

        slider.valueChanged.connect(update_value)
        container.addWidget(slider)
        
        container.addSpacing(8)
        
        return container
    
    def on_dataset_changed(self, name: str):
        if name and name != "No datasets loaded":
            self.dataset_manager.set_active(name)
            self.display_original_signal()

    def upload_dataset(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "NumPy Files (*.npy);;All Files (*)")

        if not filepath:
            return
        
        try:
            signal = np.load(filepath)

            from pathlib import Path
            name = Path(filepath).stem

            fs = 8e6
            metadata = {
                'source': 'uploaded',
                'filepath': filepath
            }

            self.dataset_manager.add_dataset(name, signal, fs, metadata)
            self.refresh_dataset_list()

            print("Uploaded dataset: {name}")

        except Exception as e:
            print(f"Failed to upload dataset: {e}")

    def refresh_dataset_list(self):
        self.dataset_combo.clear()
        datasets = self.dataset_manager.list_datasets()

        if datasets:
            self.dataset_combo.addItems(datasets)
            active = self.dataset_manager.active_dataset
            if active and active in datasets:
                self.dataset_combo.setCurrentText(active)
        else:
            self.dataset_combo.addItem("No datasets loaded")

        print("Dataset list refreshed: {datasets}")

    def display_original_signal(self):
        dataset = self.dataset_manager.get_active()
        if dataset is None:
            print("No active dataset to display")
            return
        
        self.clean_signal = dataset['signal']
        self.clean_fs = dataset['fs']
        self.clean_metadata = dataset.get('metadata', {})

    def apply_augmentations(self):
        if not hasattr(self, 'clean_signal'):
            print("No dataset selected to augment.")
            return
        
        signal = self.clean_signal
        fs = self.clean_fs

        pipeline = AugmentationPipeline()

        if self.awgn_enabled:
            pipeline.add(AWGNAugmentation(snr_db=self.snr_db))

        if self.amp_phase_enabled:
            phase_rad = np.deg2rad(self.phase_deg)
            pipeline.add(ScalarAmplitudeAndPhaseShift(amplitude=self.amplitude, phi=phase_rad))

        augmented_signal = pipeline.apply(signal, fs)

        # Get metadata for constellation plot
        metadata = self.clean_metadata
        fc = metadata.get('fc', None)
        Tsymb = metadata.get('Tsymb', None)
        modulation = metadata.get('modulation', None)
        
        # Calculate sps if available
        sps = None
        if Tsymb is not None:
            sps = int(fs * Tsymb)
        
        # Display comparison
        self.comparison_plot.plot_comparison(
            clean_signal=signal,
            augmented_signal=augmented_signal,
            fs=fs,
            fc=fc,
            sps=sps,
            modulation=modulation
        )
        
        # Store augmented signal temporarily
        self.last_augmented_signal = augmented_signal
        self.last_augmented_fs = fs
    

    def save_augmented_dataset(self):
        if not hasattr(self, 'last_augmented_signal'):
            print("No augmented signal to save")
            return
        
        active_name = self.dataset_manager.active_dataset
        if not active_name:
            active_name = "dataset"

        aug_name = f"{active_name}_augmented"
        counter = 1
        while aug_name in self.dataset_manager.datasets:
            aug_name = f"{active_name}_augmented_{counter}"
            counter += 1

        # Get original metadata
        original_dataset = self.dataset_manager.get_active()
        metadata = original_dataset['metadata'].copy() if original_dataset else {}

        # Add augmentation info to metadata
        metadata['augmented'] = True
        metadata['base_dataset'] = active_name
        metadata['augmentations'] = []
        if self.awgn_enabled:
            metadata['augmentations'].append(f"AWGN(snr={self.snr_db}dB)")
        if self.amp_phase_enabled:
            metadata['augmentations'].append(f"AmpPhase(amp={self.amplitude}, phi={self.phase_deg}deg)")

        self.dataset_manager.add_dataset(
            aug_name,
            self.last_augmented_signal,
            self.last_augmented_fs,
            metadata
        )

        self.refresh_dataset_list()
        print(f"Saved augmented dataset: {aug_name}")

    

    