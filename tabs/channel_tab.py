from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QSlider, QGridLayout)
from PySide6.QtCore import Qt

from widgets.toggle_switch import ToggleSwitch
from widgets.noise_spectrum import NoiseSpectrumWidget


class ChannelNoiseTab(QWidget):
    """Channel configuration and noise settings tab"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - Channel Configuration
        left_panel = self.create_channel_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel - Noise Configuration and Spectrum
        right_panel = self.create_noise_panel()
        layout.addWidget(right_panel, 1)
    
    def create_channel_panel(self):
        """Create the channel configuration panel"""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("📡 Channel Configuration")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Configure individual channel parameters")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        # Channel table
        table = QTableWidget(4, 4)
        table.setHorizontalHeaderLabels(["Channel", "Status", "Gain (dB)", "SNR (dB)"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(50)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setMaximumHeight(280)
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        channels_data = [
            ("Channel 1", True, "80 dB", "25.5 dB", False),
            ("Channel 2", True, "75 dB", "28.3 dB", False),
            ("Channel 3", False, "60 dB", "22.1 dB", True),
            ("Channel 4", True, "85 dB", "30.2 dB", False)
        ]
        
        for row, (channel, enabled, gain, snr, is_low_snr) in enumerate(channels_data):
            # Channel name
            channel_widget = QWidget()
            channel_layout = QHBoxLayout(channel_widget)
            channel_layout.setContentsMargins(12, 0, 8, 0)
            channel_layout.setAlignment(Qt.AlignLeft)
            channel_label = QLabel(channel)
            channel_label.setStyleSheet("font-size: 13px; color: #1f2937; font-weight: 500;")
            channel_layout.addWidget(channel_label)
            channel_layout.addStretch()
            table.setCellWidget(row, 0, channel_widget)
            
            # Status toggle
            status_widget = QWidget()
            status_layout = QHBoxLayout(status_widget)
            status_layout.setContentsMargins(8, 0, 8, 0)
            status_layout.setAlignment(Qt.AlignLeft)
            toggle = ToggleSwitch()
            toggle.setChecked(enabled)
            status_layout.addWidget(toggle)
            status_layout.addStretch()
            table.setCellWidget(row, 1, status_widget)
            
            # Gain badge
            gain_widget = QWidget()
            gain_layout = QHBoxLayout(gain_widget)
            gain_layout.setContentsMargins(8, 0, 8, 0)
            gain_layout.setAlignment(Qt.AlignLeft)
            gain_label = QLabel(gain)
            gain_label.setStyleSheet("background-color: #111827; color: white; border-radius: 12px; padding: 6px 14px; font-size: 13px; font-weight: 600;")
            gain_label.setAlignment(Qt.AlignCenter)
            gain_layout.addWidget(gain_label)
            gain_layout.addStretch()
            table.setCellWidget(row, 2, gain_widget)
            
            # SNR badge
            snr_widget = QWidget()
            snr_layout = QHBoxLayout(snr_widget)
            snr_layout.setContentsMargins(8, 0, 8, 0)
            snr_layout.setAlignment(Qt.AlignLeft)
            snr_label = QLabel(snr)
            if is_low_snr:
                snr_label.setStyleSheet("background-color: #ef4444; color: white; border-radius: 12px; padding: 6px 14px; font-size: 13px; font-weight: 600;")
            else:
                snr_label.setStyleSheet("background-color: #111827; color: white; border-radius: 12px; padding: 6px 14px; font-size: 13px; font-weight: 600;")
            snr_label.setAlignment(Qt.AlignCenter)
            snr_layout.addWidget(snr_label)
            snr_layout.addStretch()
            table.setCellWidget(row, 3, snr_widget)
        
        layout.addWidget(table)
        
        # Info box
        info_box = QFrame()
        info_box.setObjectName("infoBox")
        info_layout = QHBoxLayout(info_box)
        info_icon = QLabel("ⓘ")
        info_icon.setStyleSheet("font-size: 16px; color: #3b82f6;")
        info_text = QLabel("<b>Active Channels</b><br/>3 of 4 channels enabled")
        info_text.setProperty("class", "section-subtitle")
        info_layout.addWidget(info_icon)
        info_layout.addWidget(info_text)
        info_layout.addStretch()
        layout.addWidget(info_box)
        
        layout.addStretch()
        
        return panel
    
    def create_noise_panel(self):
        """Create the noise configuration and visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Noise Configuration Card
        noise_card = self.create_noise_config_card()
        layout.addWidget(noise_card)
        
        # Noise Power Spectrum Card
        spectrum_card = self.create_spectrum_card()
        layout.addWidget(spectrum_card)
        
        return panel
    
    def create_noise_config_card(self):
        """Create the noise configuration card"""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        title = QLabel("🔊 Noise Configuration")
        title.setProperty("class", "card-title")
        subtitle = QLabel("Configure noise and interference parameters")
        subtitle.setProperty("class", "section-subtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(2)
        header.addLayout(title_layout)
        layout.addLayout(header)
        
        # Noise Level Slider
        noise_slider_layout = self.create_slider_control("Noise Level", 30, "dBm", -50, 0)
        layout.addLayout(noise_slider_layout)
        
        # AWGN Toggle
        awgn_layout = QHBoxLayout()
        awgn_left = QVBoxLayout()
        awgn_title = QLabel("AWGN (Additive White Gaussian Noise)")
        awgn_title.setProperty("class", "section-title")
        awgn_desc = QLabel("Enable white noise generation")
        awgn_desc.setProperty("class", "section-subtitle")
        awgn_left.addWidget(awgn_title)
        awgn_left.addWidget(awgn_desc)
        awgn_layout.addLayout(awgn_left)
        awgn_layout.addStretch()
        awgn_toggle = ToggleSwitch()
        awgn_toggle.setChecked(True)
        awgn_layout.addWidget(awgn_toggle)
        layout.addLayout(awgn_layout)
        
        # Multipath Fading Toggle
        multipath_layout = QHBoxLayout()
        multipath_left = QVBoxLayout()
        multipath_title = QLabel("Multipath Fading")
        multipath_title.setProperty("class", "section-title")
        multipath_desc = QLabel("Simulate multipath interference")
        multipath_desc.setProperty("class", "section-subtitle")
        multipath_left.addWidget(multipath_title)
        multipath_left.addWidget(multipath_desc)
        multipath_layout.addLayout(multipath_left)
        multipath_layout.addStretch()
        multipath_toggle = ToggleSwitch()
        multipath_toggle.setChecked(False)
        multipath_layout.addWidget(multipath_toggle)
        layout.addLayout(multipath_layout)
        
        # Apply button
        apply_btn = QPushButton("Apply Noise Settings")
        apply_btn.setObjectName("primaryButton")
        layout.addWidget(apply_btn)
        
        return card
    
    def create_spectrum_card(self):
        """Create the noise power spectrum card"""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        title = QLabel("Noise Power Spectrum")
        title.setProperty("class", "card-title")
        subtitle = QLabel("Frequency distribution of noise components")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        # Spectrum visualization
        spectrum_widget = NoiseSpectrumWidget()
        layout.addWidget(spectrum_widget)
        
        # Stats
        stats_layout = QGridLayout()
        stats_layout.setSpacing(20)
        
        stats = [
            ("Total Noise Power", "170 dBm"),
            ("Bandwidth", "60 Hz"),
            ("Noise Figure", "5.2 dB")
        ]
        
        for i, (label, value) in enumerate(stats):
            stat_container = QVBoxLayout()
            stat_label = QLabel(label)
            stat_label.setProperty("class", "stat-label")
            stat_value = QLabel(value)
            stat_value.setProperty("class", "stat-value")
            stat_container.addWidget(stat_label)
            stat_container.addWidget(stat_value)
            stats_layout.addLayout(stat_container, 0, i)
        
        layout.addLayout(stats_layout)
        
        return card
    
    def create_slider_control(self, label, value, unit, min_val, max_val):
        """Create a slider control with label and value display"""
        container = QVBoxLayout()
        container.setSpacing(8)
        
        # Label and value
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
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
        slider.setValue(value)
        slider.valueChanged.connect(lambda v, lbl=value_label, u=unit: lbl.setText(f"{v} {u}"))
        container.addWidget(slider)
        
        container.addSpacing(8)
        
        return container