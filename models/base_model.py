"""
Base model interface for multi-modal completion.

Defines the abstract interface that all generative models (Flow Matching, DDPM, Score SDE)
must implement for modal completion tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseGenerativeModel(ABC, nn.Module):
    """
    Abstract base class for generative models supporting multi-modal completion.
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 num_modalities: int,
                 channels_per_modality: int,
                 **kwargs):
        """
        Initialize base generative model.
        
        Args:
            backbone: Neural network backbone (e.g., U-Net from diffusers)
            num_modalities: Number of modalities in the data
            channels_per_modality: Number of channels per modality
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities
        self.channels_per_modality = channels_per_modality
        self.total_channels = num_modalities * channels_per_modality
        
    @abstractmethod
    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                task_mask: torch.Tensor,
                x_condition: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x_t: Noisy input at time t, shape (B, C, H, W) or (B, C, D, H, W)
            t: Time step, shape (B,) or (B, 1)
            task_mask: Binary mask indicating which modalities to generate, shape (B, num_modalities)
            x_condition: Optional clean condition modalities
            **kwargs: Additional model-specific inputs
            
        Returns:
            Model prediction (interpretation depends on model type)
        """
        pass
    
    @abstractmethod
    def compute_loss(self,
                    x_0: torch.Tensor,
                    task_id: Optional[int] = None,
                    task_mask: Optional[torch.Tensor] = None,
                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for a batch of clean data.
        
        Args:
            x_0: Clean data, shape (B, C, H, W) or (B, C, D, H, W)
            task_id: Optional fixed task ID (if None, sample randomly)
            task_mask: Optional pre-computed task mask
            **kwargs: Additional loss parameters
            
        Returns:
            Dictionary containing:
                - 'loss': Total loss value
                - Additional metrics for logging
        """
        pass
    
    @abstractmethod
    def sample(self,
              x_condition: torch.Tensor,
              task_mask: torch.Tensor,
              num_steps: int = 50,
              **kwargs) -> torch.Tensor:
        """
        Generate samples conditioned on given modalities.
        
        Args:
            x_condition: Clean condition modalities, shape (B, C, H, W)
            task_mask: Binary mask indicating which modalities to generate, shape (B, num_modalities)
            num_steps: Number of sampling steps
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated samples, shape (B, C, H, W)
        """
        pass
    
    def prepare_input(self,
                     x_clean: torch.Tensor,
                     x_noisy: torch.Tensor,
                     task_mask: torch.Tensor) -> torch.Tensor:
        """
        Prepare model input by combining clean and noisy modalities based on task mask.
        
        Args:
            x_clean: Clean data
            x_noisy: Noisy data
            task_mask: Binary mask
            
        Returns:
            Combined input tensor
        """
        from ..utils.modal_utils import combine_modalities
        return combine_modalities(x_clean, x_noisy, task_mask)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving/loading.
        
        Returns:
            Configuration dictionary
        """
        return {
            'num_modalities': self.num_modalities,
            'channels_per_modality': self.channels_per_modality,
            'total_channels': self.total_channels,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], backbone: nn.Module):
        """
        Create model instance from configuration.
        
        Args:
            config: Configuration dictionary
            backbone: Backbone network
            
        Returns:
            Model instance
        """
        return cls(backbone=backbone, **config)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    """
    
    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embeddings.
        
        Args:
            t: Time steps, shape (B,) or (B, 1), values in [0, 1]
            
        Returns:
            Time embeddings, shape (B, embedding_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
            
        half_dim = self.embedding_dim // 2
        embeddings = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * 
            torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


class TaskEmbedding(nn.Module):
    """
    Embedding module for task masks.
    """
    
    def __init__(self, num_modalities: int, embedding_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.embedding_dim = embedding_dim
        self.embed = nn.Linear(num_modalities, embedding_dim)
        
    def forward(self, task_mask: torch.Tensor) -> torch.Tensor:
        """
        Embed task mask into continuous representation.
        
        Args:
            task_mask: Binary task mask, shape (B, num_modalities)
            
        Returns:
            Task embeddings, shape (B, embedding_dim)
        """
        return self.embed(task_mask.float())
