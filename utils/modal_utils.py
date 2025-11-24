"""
Utilities for multi-modal completion tasks.

This module provides functions to encode/decode generation tasks using binary representation,
where each bit indicates whether a modality is a condition (0) or target for generation (1).
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union


def task_to_binary_mask(task_id: int, num_modalities: int) -> torch.Tensor:
    """
    Convert task ID to binary mask indicating which modalities to generate.
    
    Args:
        task_id: Integer representing the task (0 to 2^num_modalities - 1)
        num_modalities: Total number of modalities
        
    Returns:
        Binary tensor of shape (num_modalities,) where 1 indicates generation target
        
    Example:
        >>> task_to_binary_mask(5, 4)  # 5 = 0b0101
        tensor([1, 0, 1, 0])  # Generate modalities 0 and 2
    """
    mask = torch.zeros(num_modalities, dtype=torch.bool)
    for i in range(num_modalities):
        if task_id & (1 << i):
            mask[i] = True
    return mask


def binary_mask_to_task(mask: torch.Tensor) -> int:
    """
    Convert binary mask to task ID.
    
    Args:
        mask: Binary tensor of shape (num_modalities,)
        
    Returns:
        Integer task ID
    """
    task_id = 0
    if mask.dim() > 1:
        mask = mask[0]
    for i, val in enumerate(mask):
        if val:
            task_id |= (1 << i)
    return task_id


def sample_random_task(num_modalities: int, 
                      exclude_empty: bool = True,
                      exclude_full: bool = False) -> int:
    """
    Sample a random generation task.
    
    Args:
        num_modalities: Total number of modalities
        exclude_empty: If True, exclude task 0 (pure generation, no conditions)
        exclude_full: If True, exclude task 2^n-1 (all modalities as conditions)
        
    Returns:
        Random task ID
    """
    min_task = 1 if exclude_empty else 0
    max_task = (2 ** num_modalities) - (2 if exclude_full else 1)
    return np.random.randint(min_task, max_task + 1)


def combine_modalities(x_clean: torch.Tensor,
                       x_noisy: torch.Tensor,
                       task_mask: torch.Tensor,
                       channel_dim: int = 1,
                       channels_per_modality: Optional[Union[int, list[int], torch.Tensor]] = None,) -> torch.Tensor:
    """
    Combine clean (condition) and noisy (generation target) modalities based on task mask.
    
    Args:
        x_clean: Clean data tensor, shape (B, C, H, W) or (B, C, D, H, W)
        x_noisy: Noisy data tensor, same shape as x_clean
        task_mask: Binary mask, shape (num_modalities,) or (B, num_modalities)
        channel_dim: Dimension along which modalities are concatenated
        
    Returns:
        Combined tensor where condition modalities use x_clean and target modalities use x_noisy
    """
    if task_mask.dim() == 1:
        # Single task mask for entire batch
        task_mask = task_mask.unsqueeze(0)
    
    # Reshape mask to match data dimensions
    batch_size = x_clean.shape[0]
    num_modalities = task_mask.shape[-1]
    
    # Expand mask to match spatial dimensions
    mask_shape = [1] * x_clean.dim()
    mask_shape[0] = batch_size
    mask_shape[channel_dim] = num_modalities
    expanded_mask = task_mask.view(*mask_shape)
    
    # Repeat mask for each channel within modality
    if channels_per_modality is None:
        channels_per_modality = x_clean.shape[channel_dim] // num_modalities
    if isinstance(channels_per_modality, list):
        channels_per_modality = torch.tensor(channels_per_modality, device=expanded_mask.device)
    expanded_mask = expanded_mask.repeat_interleave(channels_per_modality, dim=channel_dim)
    
    # Expand to all spatial dimensions
    """for dim in range(channel_dim + 1, x_clean.dim()):
        expanded_mask = expanded_mask.unsqueeze(dim)"""
    expanded_mask = expanded_mask.expand_as(x_clean)
    
    # Combine: use x_noisy where mask=1 (generate), x_clean where mask=0 (condition)
    combined = torch.where(expanded_mask, x_noisy, x_clean)
    return combined


def get_generation_mask(task_id: int, 
                       num_modalities: int,
                       channels_per_modality: Union[int, list[int], torch.Tensor],
                       spatial_shape: Tuple[int, ...],
                       device: torch.device) -> torch.Tensor:
    """
    Create a full mask tensor for identifying which parts need generation.
    
    Args:
        task_id: Task ID
        num_modalities: Number of modalities
        channels_per_modality: Channels per modality
        spatial_shape: Spatial dimensions (H, W) or (D, H, W)
        device: Target device
        
    Returns:
        Mask tensor of shape (1, C, *spatial_shape)
    """
    task_mask = task_to_binary_mask(task_id, num_modalities)
    
    # Expand to channel dimension
    if isinstance(channels_per_modality, list):
        channels_per_modality = torch.tensor(channels_per_modality, device=task_mask.device)
    channel_mask = task_mask.repeat_interleave(channels_per_modality)
    
    # Expand to spatial dimensions
    mask = channel_mask.view(1, -1, *([1] * len(spatial_shape)))
    mask = mask.expand(1, -1, *spatial_shape)
    
    return mask.to(device)


def get_task_description(task_id: int, 
                        num_modalities: int,
                        modality_names: Optional[List[str]] = None) -> str:
    """
    Get human-readable description of a task.
    
    Args:
        task_id: Task ID
        num_modalities: Number of modalities
        modality_names: Optional list of modality names
        
    Returns:
        Task description string
    """
    if modality_names is None:
        modality_names = [f"M{i}" for i in range(num_modalities)]
    
    mask = task_to_binary_mask(task_id, num_modalities)
    
    conditions = [modality_names[i] for i, m in enumerate(mask) if not m]
    targets = [modality_names[i] for i, m in enumerate(mask) if m]
    
    if not conditions:
        return f"Unconditional generation of {', '.join(targets)}"
    elif not targets:
        return "No generation (all modalities are conditions)"
    else:
        return f"Generate {', '.join(targets)} conditioned on {', '.join(conditions)}"
