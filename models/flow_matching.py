"""
Flow Matching model for multi-modal completion.

Implements Conditional Flow Matching (CFM) with multiple interpolation paths.
Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union
from .base_model import BaseGenerativeModel
from utils.modal_utils import (
    task_to_binary_mask, 
    binary_mask_to_task,
    sample_random_task,
    combine_modalities
)


class FlowMatchingModel(BaseGenerativeModel):
    """
    Flow Matching model with support for multi-modal conditional generation.
    
    Supports multiple interpolation schemes including linear OT and variance-preserving paths.
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 num_modalities: int,
                 channels_per_modality: Union[int, list[int], torch.Tensor],
                 sigma_min: float = 1e-4,
                 path_type: str = 'linear',
                 **kwargs):
        """
        Initialize Flow Matching model.
        
        Args:
            backbone: Neural network backbone
            num_modalities: Number of modalities
            channels_per_modality: Channels per modality
            sigma_min: Minimum noise level for numerical stability
            path_type: Type of interpolation path ('linear', 'vp', 'vp_simple')
                - 'linear': x_t = (1-t)*x_1 + t*x_0, u_t = x_0 - x_1
                - 'vp': variance-preserving path similar to DDPM
                - 'vp_simple': simplified VP with sigma_min
            **kwargs: Additional parameters
        """
        super().__init__(backbone, num_modalities, channels_per_modality, **kwargs)
        self.sigma_min = sigma_min
        self.path_type = path_type
        
        # Check if backbone supports class embedding
        self.use_class_embed = hasattr(backbone, 'class_embedding') or \
                                hasattr(backbone, 'class_embed') or \
                                (hasattr(backbone, 'config') and 
                                 hasattr(backbone.config, 'num_class_embeds') and
                                 backbone.config.num_class_embeds is not None)
        
    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                task_mask: torch.Tensor,
                task_id: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """
        Predict the flow (velocity field) at time t.
        
        Args:
            x_t: State at time t
            t: Time in [0, 1]
            task_mask: Task mask indicating condition/generation modalities
            task_id: Task ID for class embedding (if backbone supports it)
            **kwargs: Additional arguments
            
        Returns:
            Predicted velocity field
        """
        # Ensure t is in correct shape for diffusers models
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # Prepare backbone input arguments
        backbone_kwargs = {}
        
        # If backbone supports class embedding, pass task_id as class label
        if self.use_class_embed:
            if task_id is None:
                task_id = binary_mask_to_task(task_mask)
            if isinstance(task_id, int):
                batch_size = x_t.shape[0]
                class_labels = torch.full((batch_size,), task_id, 
                                         device=x_t.device, dtype=torch.long)
            else:
                class_labels = task_id
            backbone_kwargs['class_labels'] = class_labels
        
        # Call backbone
        output = self.backbone(x_t, t, **backbone_kwargs)
        
        if hasattr(output, 'sample'):
            prediction = output.sample
        else:
            prediction = output
            
        return prediction
    
    def compute_loss(self,
                    x_0: torch.Tensor,
                    task_id: Optional[int] = None,
                    task_mask: Optional[torch.Tensor] = None,
                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute Flow Matching loss.
        
        The loss is: L = E[||v_θ(x_t, t) - u_t||^2]
        where u_t is the conditional velocity field.
        
        Args:
            x_0: Clean data
            task_id: Optional task ID
            task_mask: Optional task mask
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'loss' and other metrics
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample or use provided task mask
        if task_mask is None:
            if task_id is None:
                task_id = sample_random_task(self.num_modalities, exclude_empty=False)
            task_mask = task_to_binary_mask(task_id, self.num_modalities)
            task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        x_1 = torch.randn_like(x_0)
        
        # Compute interpolated state and velocity based on path type
        t_expanded = t.view(batch_size, *([1] * (x_0.dim() - 1)))
        
        if self.path_type == 'linear':
            # Standard linear interpolation
            x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
            u_t = x_0 - x_1
            
        elif self.path_type == 'vp_simple':
            # Simplified variance-preserving with sigma_min
            sigma_t = 1 - (1 - self.sigma_min) * t_expanded
            x_t = t_expanded * x_0 + sigma_t * x_1
            u_t = x_0 - (1 - self.sigma_min) * x_1
            
        elif self.path_type == 'vp':
            # Variance-preserving path (similar to DDPM)
            alpha_t = torch.cos(t_expanded * torch.pi / 2)
            sigma_t = torch.sin(t_expanded * torch.pi / 2)
            x_t = alpha_t * x_0 + sigma_t * x_1
            u_t = -torch.pi / 2 * (torch.sin(t_expanded * torch.pi / 2) * x_0 - 
                                    torch.cos(t_expanded * torch.pi / 2) * x_1)
        else:
            raise ValueError(f"Unknown path_type: {self.path_type}")
        
        # Combine clean and noisy data based on task mask
        x_input = combine_modalities(x_0, x_t, task_mask, channels_per_modality=self.channels_per_modality)
        
        # Predict velocity
        v_pred = self.forward(x_input, t, task_mask, task_id=task_id)
        
        # Compute loss only on generation modalities
        loss_mask = task_mask.float()
        if isinstance(self.channels_per_modality, list):
            repeats = torch.tensor(self.channels_per_modality, device=loss_mask.device)
            loss_mask = loss_mask.repeat_interleave(repeats, dim=1)
        else:
            loss_mask = loss_mask.repeat_interleave(self.channels_per_modality, dim=1)
        for _ in range(x_0.dim() - 2):
            loss_mask = loss_mask.unsqueeze(-1)
        loss_mask = loss_mask.expand_as(x_0)
        
        # MSE loss weighted by task mask
        mse_loss = ((v_pred - u_t) ** 2) * loss_mask
        loss = mse_loss.sum() / (loss_mask.sum() + 1e-8)
        
        return {
            'loss': loss,
            'mse': loss.item(),
        }
    
    @torch.no_grad()
    def sample(self,
              x_condition: torch.Tensor,
              task_mask: torch.Tensor,
              task_id: Optional[int] = None,
              num_steps: int = 50,
              method: str = 'euler',
              **kwargs) -> torch.Tensor:
        """
        Generate samples using ODE integration.
        
        Args:
            x_condition: Condition modalities
            task_mask: Task mask
            task_id: Task ID for class embedding
            num_steps: Number of integration steps
            method: Integration method ('euler', 'heun', 'rk4')
            **kwargs: Additional parameters
            
        Returns:
            Generated samples
        """
        batch_size = x_condition.shape[0]
        device = x_condition.device
        
        # Start from noise
        x_t = torch.randn_like(x_condition)
        x_t = combine_modalities(x_condition, x_t, task_mask, channels_per_modality=self.channels_per_modality)
        
        # Time steps
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            if method == 'euler':
                # Euler method
                v_t = self.forward(x_t, t, task_mask, task_id=task_id)
                x_t_next = x_t + dt * v_t
                
            elif method == 'heun':
                # Heun's method (2nd order)
                v_t = self.forward(x_t, t, task_mask, task_id=task_id)
                x_t_pred = x_t + dt * v_t
                
                t_next = t + dt
                v_t_next = self.forward(x_t_pred, t_next, task_mask, task_id=task_id)
                x_t_next = x_t + dt * (v_t + v_t_next) / 2
                
            elif method == 'rk4':
                # 4th order Runge-Kutta
                t_half = t + dt / 2
                t_next = t + dt
                
                k1 = self.forward(x_t, t, task_mask, task_id=task_id)
                k2 = self.forward(x_t + dt * k1 / 2, t_half, task_mask, task_id=task_id)
                k3 = self.forward(x_t + dt * k2 / 2, t_half, task_mask, task_id=task_id)
                k4 = self.forward(x_t + dt * k3, t_next, task_mask, task_id=task_id)
                
                x_t_next = x_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Keep condition modalities unchanged
            x_t = combine_modalities(x_condition, x_t_next, task_mask, channels_per_modality=self.channels_per_modality)
        
        return x_t
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'sigma_min': self.sigma_min,
            'path_type': self.path_type,
        })
        return config