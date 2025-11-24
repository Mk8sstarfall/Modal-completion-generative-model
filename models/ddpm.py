"""
DDPM (Denoising Diffusion Probabilistic Model) for multi-modal completion.

Implements DDPM as a special case within the unified framework.
Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from .base_model import BaseGenerativeModel
from utils.modal_utils import (
    task_to_binary_mask,
    binary_mask_to_task,
    sample_random_task,
    combine_modalities
)


class DDPMModel(BaseGenerativeModel):
    """
    DDPM model with support for multi-modal conditional generation.
    
    Uses linear or cosine noise schedule for the forward diffusion process.
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 num_modalities: int,
                 channels_per_modality: Union[int, list[int], torch.Tensor],
                 num_train_timesteps: int = 1000,
                 beta_schedule: str = 'linear',
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 prediction_type: str = 'epsilon',
                 **kwargs):
        """
        Initialize DDPM model.
        
        Args:
            backbone: Neural network backbone
            num_modalities: Number of modalities
            channels_per_modality: Channels per modality
            num_train_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type ('linear', 'cosine', 'scaled_linear')
            beta_start: Starting beta value
            beta_end: Ending beta value
            prediction_type: Type of prediction ('epsilon', 'sample', 'v_prediction')
            **kwargs: Additional parameters
        """
        super().__init__(backbone, num_modalities, channels_per_modality, **kwargs)
        
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        
        # Check if backbone supports class embedding
        self.use_class_embed = hasattr(backbone, 'class_embedding') or \
                                hasattr(backbone, 'class_embed') or \
                                (hasattr(backbone, 'config') and 
                                 hasattr(backbone.config, 'num_class_embeds') and
                                 backbone.config.num_class_embeds is not None)
        
        # Create noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            # Used in Stable Diffusion
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract values from a 1-D tensor for a batch of indices.
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process: sample x_t from q(x_t | x_0).
        
        Args:
            x_0: Clean data
            t: Timestep indices
            noise: Gaussian noise
            
        Returns:
            Noisy samples x_t
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                task_mask: torch.Tensor,
                task_id: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        """
        Predict noise or other target based on prediction_type.
        
        Args:
            x_t: Noisy input at timestep t
            t: Timestep (discrete, in [0, num_train_timesteps))
            task_mask: Task mask indicating condition/generation modalities
            task_id: Task ID for class embedding (if backbone supports it)
            **kwargs: Additional arguments
            
        Returns:
            Model prediction (noise, sample, or v-prediction)
        """
        # Prepare backbone input arguments
        backbone_kwargs = {}
        
        # If backbone supports class embedding, pass task_id as class label
        if self.use_class_embed:
            if task_id is None:
                task_id = binary_mask_to_task(task_mask)
            if isinstance(task_id, int):
                # Convert single task_id to batch
                batch_size = x_t.shape[0]
                class_labels = torch.full((batch_size,), task_id, 
                                         device=x_t.device, dtype=torch.long)
            else:
                class_labels = task_id
            backbone_kwargs['class_labels'] = class_labels
        
        # Call backbone with appropriate arguments
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
        Compute DDPM training loss.
        
        Args:
            x_0: Clean data
            task_id: Optional task ID
            task_mask: Optional task mask
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'loss' and metrics
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample or use provided task mask
        if task_mask is None:
            if task_id is None:
                task_id = sample_random_task(self.num_modalities, exclude_empty=False)
            task_mask = task_to_binary_mask(task_id, self.num_modalities)
            task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Sample timesteps uniformly
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Combine clean and noisy data based on task mask
        x_input = combine_modalities(x_0, x_t, task_mask, channels_per_modality=self.channels_per_modality)
        
        # Predict
        prediction = self.forward(x_input, t, task_mask, task_id=task_id)
        
        # Compute target based on prediction type
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = x_0
        elif self.prediction_type == 'v_prediction':
            # v = sqrt(alpha_t) * noise - sqrt(1-alpha_t) * x_0
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
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
        mse_loss = ((prediction - target) ** 2) * loss_mask
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
              num_steps: Optional[int] = None,
              eta: float = 0.0,
              **kwargs) -> torch.Tensor:
        """
        Generate samples using DDPM or DDIM sampling.
        
        Args:
            x_condition: Condition modalities
            task_mask: Task mask
            task_id: Task ID for class embedding
            num_steps: Number of sampling steps (if None, use all timesteps)
            eta: DDIM parameter (0 for deterministic, 1 for DDPM)
            **kwargs: Additional parameters
            
        Returns:
            Generated samples
        """
        batch_size = x_condition.shape[0]
        device = x_condition.device
        
        if num_steps is None:
            num_steps = self.num_train_timesteps
            timesteps = list(range(self.num_train_timesteps))[::-1]
        else:
            # DDIM-style timestep subsampling
            step_ratio = self.num_train_timesteps // num_steps
            timesteps = list(range(0, self.num_train_timesteps, step_ratio))[::-1]
        
        # Start from noise
        x_t = torch.randn_like(x_condition)
        x_t = combine_modalities(x_condition, x_t, task_mask, channels_per_modality=self.channels_per_modality)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x_t, t_batch, task_mask, task_id=task_id)
            
            # Compute predicted x_0
            alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            if self.prediction_type == 'epsilon':
                pred_x_0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            elif self.prediction_type == 'sample':
                pred_x_0 = predicted_noise
            elif self.prediction_type == 'v_prediction':
                pred_x_0 = sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * predicted_noise
            
            # Clip predicted x_0
            pred_x_0 = torch.clamp(pred_x_0, -1, 1)
            
            # Compute x_{t-1}
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev_batch, x_t.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)
            
            # DDIM sampling
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t)) * predicted_noise
            
            if eta > 0:
                noise = torch.randn_like(x_t)
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                x_t_prev = sqrt_alpha_t_prev * pred_x_0 + dir_xt + sigma_t * noise
            else:
                x_t_prev = sqrt_alpha_t_prev * pred_x_0 + dir_xt
            
            # Keep condition modalities unchanged
            x_t = combine_modalities(x_condition, x_t_prev, task_mask, channels_per_modality=self.channels_per_modality)
        
        return x_t
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'num_train_timesteps': self.num_train_timesteps,
            'beta_schedule': self.beta_schedule,
            'prediction_type': self.prediction_type,
        })
        return config