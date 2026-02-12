"""
Dataset Manager - Centralized dataset storage and management
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class DatasetManager:
    """Manages datasets across the application"""
    
    def __init__(self):
        self.datasets: Dict[str, dict] = {}
        self.active_dataset: Optional[str] = None
    
    def add_dataset(self, name: str, signal: np.ndarray, fs: float, metadata: Optional[dict] = None):
        """
        Add a new dataset to the manager
        
        Args:
            name: Unique identifier for the dataset
            signal: Signal array (real or complex)
            fs: Sampling frequency in Hz
            metadata: Optional metadata (modulation type, SNR, etc.)
        """
        if metadata is None:
            metadata = {}
        
        self.datasets[name] = {
            'signal': signal.copy(),
            'fs': fs,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        self.active_dataset = name
        print(f"✓ Added dataset '{name}' ({len(signal)} samples, fs={fs/1e6:.1f} MHz)")
    
    def get_dataset(self, name: str) -> Optional[dict]:
        """Retrieve a specific dataset by name"""
        return self.datasets.get(name)
    
    def get_active(self) -> Optional[dict]:
        """Get the currently active dataset"""
        if self.active_dataset and self.active_dataset in self.datasets:
            return self.datasets[self.active_dataset]
        return None
    
    def set_active(self, name: str) -> bool:
        """Set a dataset as active"""
        if name in self.datasets:
            self.active_dataset = name
            print(f"✓ Active dataset: '{name}'")
            return True
        return False
    
    def list_datasets(self) -> List[str]:
        """List all available dataset names"""
        return list(self.datasets.keys())
    
    def remove_dataset(self, name: str) -> bool:
        """Remove a dataset"""
        if name in self.datasets:
            del self.datasets[name]
            if self.active_dataset == name:
                self.active_dataset = None
            return True
        return False
    
    def load_from_npy(self, filepath: str, name: Optional[str] = None, fs: float = 1.0, metadata: Optional[dict] = None) -> str:
        """
        Load dataset from .npy file
        
        Args:
            filepath: Path to .npy file
            name: Optional name (defaults to filename)
            fs: Sampling frequency
            metadata: Optional metadata
            
        Returns:
            Name of loaded dataset
        """
        signal = np.load(filepath)
        
        if name is None:
            name = Path(filepath).stem
        
        self.add_dataset(name, signal, fs, metadata)
        return name
    
    def save_to_npy(self, name: str, filepath: str):
        """Save dataset to .npy file"""
        dataset = self.get_dataset(name)
        if dataset is None:
            raise ValueError(f"Dataset '{name}' not found")
        
        np.save(filepath, dataset['signal'])
        print(f"✓ Saved dataset '{name}' to {filepath}")
    
    def load_config(self, filepath: str) -> dict:
        """Load dataset configuration from JSON file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    
    def save_config(self, name: str, filepath: str):
        """Save dataset metadata to JSON config file"""
        dataset = self.get_dataset(name)
        if dataset is None:
            raise ValueError(f"Dataset '{name}' not found")
        
        config = {
            'name': name,
            'fs': dataset['fs'],
            'metadata': dataset['metadata'],
            'timestamp': dataset['timestamp'],
            'signal_shape': dataset['signal'].shape,
            'signal_dtype': str(dataset['signal'].dtype)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved config for '{name}' to {filepath}")
    
    def get_info(self, name: str) -> Optional[dict]:
        """Get summary information about a dataset"""
        dataset = self.get_dataset(name)
        if dataset is None:
            return None
        
        signal = dataset['signal']
        return {
            'name': name,
            'samples': len(signal),
            'duration_ms': len(signal) / dataset['fs'] * 1000,
            'fs_mhz': dataset['fs'] / 1e6,
            'dtype': str(signal.dtype),
            'is_complex': np.iscomplexobj(signal),
            'metadata': dataset['metadata'],
            'timestamp': dataset['timestamp']
        }