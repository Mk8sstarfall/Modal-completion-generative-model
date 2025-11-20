"""
Base dataset class for multi-modal completion tasks.

Provides a template for loading and preprocessing multi-modal data.
Users should inherit from this class and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import torch
from torch.utils.data import Dataset
import numpy as np


class BaseModalDataset(Dataset, ABC):
    """
    Abstract base class for multi-modal datasets.
    
    Subclasses must implement:
    - __len__: Return the total number of samples
    - _load_sample: Load a single sample from disk/memory
    - get_modality_info: Return information about modalities
    """
    
    def __init__(self,
                 data_root: str,
                 num_modalities: int,
                 channels_per_modality: int,
                 transform: Optional[callable] = None,
                 normalize: bool = True,
                 normalization_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize base dataset.
        
        Args:
            data_root: Root directory containing data
            num_modalities: Number of modalities in the data
            channels_per_modality: Number of channels per modality
            transform: Optional transform to apply to samples
            normalize: Whether to normalize data
            normalization_range: Target range for normalization
        """
        self.data_root = data_root
        self.num_modalities = num_modalities
        self.channels_per_modality = channels_per_modality
        self.total_channels = num_modalities * channels_per_modality
        self.transform = transform
        self.normalize = normalize
        self.normalization_range = normalization_range
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass
    
    @abstractmethod
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample from storage.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'data': Concatenated multi-modal data, shape (C, H, W) or (C, D, H, W)
                - 'modalities': List of individual modality tensors (optional)
                - Additional metadata (optional)
        """
        pass
    
    @abstractmethod
    def get_modality_info(self) -> Dict[str, any]:
        """
        Get information about modalities in the dataset.
        
        Returns:
            Dictionary containing:
                - 'names': List of modality names
                - 'channels': List of channels per modality
                - Additional information
        """
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        sample = self._load_sample(idx)
        
        # Apply transforms if provided
        if self.transform is not None:
            sample['data'] = self.transform(sample['data'])
        
        # Normalize if requested
        if self.normalize:
            sample['data'] = self._normalize(sample['data'])
        
        return sample
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to target range.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Assume input is in [0, 1] or similar range
        x_min, x_max = x.min(), x.max()
        
        if x_max - x_min > 1e-6:  # Avoid division by zero
            x = (x - x_min) / (x_max - x_min)  # Scale to [0, 1]
        
        # Scale to target range
        target_min, target_max = self.normalization_range
        x = x * (target_max - target_min) + target_min
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data from target range back to [0, 1].
        
        Args:
            x: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        target_min, target_max = self.normalization_range
        x = (x - target_min) / (target_max - target_min)
        return x
    
    def split_modalities(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split concatenated multi-modal data into individual modalities.
        
        Args:
            data: Concatenated data, shape (C, H, W) or (B, C, H, W)
            
        Returns:
            List of modality tensors
        """
        channel_dim = 0 if data.dim() == 3 else 1
        modalities = torch.split(data, self.channels_per_modality, dim=channel_dim)
        return list(modalities)
    
    def combine_modalities(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine individual modality tensors into concatenated data.
        
        Args:
            modalities: List of modality tensors
            
        Returns:
            Concatenated tensor
        """
        channel_dim = 0 if modalities[0].dim() == 3 else 1
        return torch.cat(modalities, dim=channel_dim)
    
    def visualize_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a sample for visualization (denormalized).
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with visualization data
        """
        sample = self[idx]
        data = sample['data']
        
        if self.normalize:
            data = self._denormalize(data)
        
        # Convert to numpy and split modalities
        data_np = data.numpy() if isinstance(data, torch.Tensor) else data
        modalities = self.split_modalities(torch.from_numpy(data_np) if isinstance(data_np, np.ndarray) else data)
        
        modality_info = self.get_modality_info()
        modality_names = modality_info.get('names', [f'Modality_{i}' for i in range(self.num_modalities)])
        
        return {
            name: mod.numpy() if isinstance(mod, torch.Tensor) else mod
            for name, mod in zip(modality_names, modalities)
        }


class SyntheticModalDataset(BaseModalDataset):
    """
    Synthetic dataset for testing and development.
    
    Generates random multi-modal data on the fly.
    """
    
    def __init__(self,
                 num_samples: int = 1000,
                 num_modalities: int = 4,
                 channels_per_modality: int = 1,
                 image_size: int = 64,
                 **kwargs):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_modalities: Number of modalities
            channels_per_modality: Channels per modality
            image_size: Size of generated images (square)
            **kwargs: Additional arguments for BaseModalDataset
        """
        super().__init__(
            data_root='synthetic',
            num_modalities=num_modalities,
            channels_per_modality=channels_per_modality,
            **kwargs
        )
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self) -> int:
        return self.num_samples
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a random synthetic sample.
        
        Args:
            idx: Sample index (used as seed)
            
        Returns:
            Dictionary with synthetic data
        """
        # Set seed for reproducibility
        rng = np.random.RandomState(idx)
        
        # Generate random data for each modality
        modalities = []
        for i in range(self.num_modalities):
            # Create correlated patterns between modalities
            base_pattern = rng.randn(self.image_size, self.image_size)
            modality_data = np.zeros((self.channels_per_modality, self.image_size, self.image_size))
            
            for c in range(self.channels_per_modality):
                # Add some structure with Gaussian blobs
                modality_data[c] = base_pattern + rng.randn(self.image_size, self.image_size) * 0.3
        
            modalities.append(torch.from_numpy(modality_data).float())
        
        # Concatenate all modalities
        data = torch.cat(modalities, dim=0)
        
        return {
            'data': data,
            'modalities': modalities,
            'idx': idx,
        }
    
    def get_modality_info(self) -> Dict[str, any]:
        """Get information about synthetic modalities."""
        return {
            'names': [f'Modality_{i}' for i in range(self.num_modalities)],
            'channels': [self.channels_per_modality] * self.num_modalities,
            'type': 'synthetic',
        }
