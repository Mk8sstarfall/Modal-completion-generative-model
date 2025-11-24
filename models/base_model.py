"""
Base model interface for multi-modal completion.

Defines the abstract interface that all generative models (Flow Matching, DDPM, Score SDE)
must implement for modal completion tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn


class BaseGenerativeModel(ABC, nn.Module):
    
    def __init__(self,
                 backbone: nn.Module,
                 num_modalities: int,
                 channels_per_modality: Union[int, List[int], torch.Tensor],
                 vae: Optional[nn.Module] = None,
                 data_channels_per_modality: Optional[Union[int, List[int]]] = None,
                 **kwargs):
        """
        Args:
            backbone: Neural network backbone
            num_modalities: Number of modalities
            channels_per_modality: Channels per modality in MODEL's operating space (latent if VAE used)
            vae: Optional VAE for latent diffusion
            data_channels_per_modality: Channels in data space (required if VAE is used)
        """
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities
        self.channels_per_modality = channels_per_modality  # model/latent space
        self.vae = vae
        
        # calculate total channels of model space
        self.total_channels = self._compute_total_channels(channels_per_modality)
        
        if vae is not None:
            if data_channels_per_modality is None:
                raise ValueError("data_channels_per_modality required when using VAE")
            self.data_channels_per_modality = data_channels_per_modality
            self.total_data_channels = self._compute_total_channels(data_channels_per_modality)
            self.use_latent_diffusion = True
        else:
            self.data_channels_per_modality = channels_per_modality
            self.total_data_channels = self.total_channels
            self.use_latent_diffusion = False
    
    def _compute_total_channels(self, channels_per_modality) -> int:
        if isinstance(channels_per_modality, int):
            return self.num_modalities * channels_per_modality
        elif isinstance(channels_per_modality, list):
            return sum(channels_per_modality)
        else:  # torch.Tensor
            return channels_per_modality.sum().item()
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode data to latent space if VAE is available."""
        if self.vae is not None:
            return self.vae.encode(x)
        return x
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space if VAE is available."""
        if self.vae is not None:
            return self.vae.decode(z)
        return z
    
    def get_config(self) -> Dict[str, Any]:
        config = {
            'num_modalities': self.num_modalities,
            'channels_per_modality': self._to_serializable(self.channels_per_modality),
            'total_channels': self.total_channels,
            'use_latent_diffusion': self.use_latent_diffusion,
        }
        if self.use_latent_diffusion:
            config['data_channels_per_modality'] = self._to_serializable(self.data_channels_per_modality)
            config['total_data_channels'] = self.total_data_channels
        return config
    
    def _to_serializable(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return value