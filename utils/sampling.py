"""
Sampling utilities for multi-modal completion.

Provides functions for generating samples from trained models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from models.base_model import BaseGenerativeModel
from utils.modal_utils import task_to_binary_mask, get_task_description


@torch.no_grad()
def sample_conditional(
    model: BaseGenerativeModel,
    x_condition: torch.Tensor,
    task_mask: torch.Tensor,
    num_steps: int = 50,
    device: str = 'cuda',
    **kwargs
) -> torch.Tensor:
    """
    Sample from model conditioned on given modalities.
    
    Args:
        model: Trained model
        x_condition: Condition modalities, shape (B, C, H, W)
        task_mask: Binary mask indicating which modalities to generate
        num_steps: Number of sampling steps
        device: Device to use
        **kwargs: Additional sampling parameters
        
    Returns:
        Generated samples
    """
    model.eval()
    x_condition = x_condition.to(device)
    task_mask = task_mask.to(device)
    
    samples = model.sample(
        x_condition=x_condition,
        task_mask=task_mask,
        num_steps=num_steps,
        **kwargs
    )
    
    return samples


@torch.no_grad()
def sample_multiple_tasks(
    model: BaseGenerativeModel,
    x_full: torch.Tensor,
    task_ids: List[int],
    num_steps: int = 50,
    device: str = 'cuda',
    **kwargs
) -> Dict[int, torch.Tensor]:
    """
    Generate samples for multiple tasks using the same full input.
    
    Args:
        model: Trained model
        x_full: Full multi-modal data, shape (B, C, H, W)
        task_ids: List of task IDs to evaluate
        num_steps: Number of sampling steps
        device: Device to use
        **kwargs: Additional sampling parameters
        
    Returns:
        Dictionary mapping task_id to generated samples
    """
    model.eval()
    x_full = x_full.to(device)
    batch_size = x_full.shape[0]
    
    results = {}
    
    for task_id in task_ids:
        # Create task mask
        task_mask = task_to_binary_mask(task_id, model.num_modalities)
        task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Generate samples
        samples = model.sample(
            x_condition=x_full,
            task_mask=task_mask,
            num_steps=num_steps,
            **kwargs
        )
        
        results[task_id] = samples
    
    return results


def visualize_samples(
    original: torch.Tensor,
    generated: torch.Tensor,
    task_mask: torch.Tensor,
    num_modalities: int,
    channels_per_modality: Union[int, List[int]],
    save_path: Optional[Path] = None,
    modality_names: Optional[List[str]] = None,
    task_id: Optional[int] = None,
) -> plt.Figure:
    """
    Visualize original and generated modalities side by side.
    
    Args:
        original: Original data, shape (C, H, W)
        generated: Generated data, shape (C, H, W)
        task_mask: Task mask, shape (num_modalities,)
        num_modalities: Number of modalities
        channels_per_modality: Channels per modality
        save_path: Optional path to save figure
        modality_names: Optional list of modality names
        task_id: Optional task ID for title
        
    Returns:
        Matplotlib figure
    """
    if modality_names is None:
        modality_names = [f'M{i}' for i in range(num_modalities)]
    
    # Convert to numpy and denormalize
    original_np = original.cpu().numpy()
    generated_np = generated.cpu().numpy()
    
    # Normalize to [0, 1] for visualization
    original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min() + 1e-8)
    generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min() + 1e-8)
    
    # Split into modalities
    if isinstance(channels_per_modality, int):
        original_mods = np.split(original_np, num_modalities, axis=0)
        generated_mods = np.split(generated_np, num_modalities, axis=0)
    else:
        split_indices = np.cumsum(channels_per_modality)[:-1]
        original_mods = np.split(original_np, split_indices, axis=0)
        generated_mods = np.split(generated_np, split_indices, axis=0)
    task_mask_np = task_mask.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, num_modalities, figsize=(4 * num_modalities, 8))
    
    if num_modalities == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_modalities):
        # Display first channel of each modality
        orig_img = original_mods[i][0]
        gen_img = generated_mods[i][0]
        
        # Original
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'{modality_names[i]} (Original)')
        axes[0, i].axis('off')
        
        # Generated
        axes[1, i].imshow(gen_img, cmap='gray')
        status = 'Generated' if task_mask_np[i] else 'Condition'
        axes[1, i].set_title(f'{modality_names[i]} ({status})')
        axes[1, i].axis('off')
        
        # Add border to generated modalities
        if task_mask_np[i]:
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    # Add overall title
    if task_id is not None:
        task_desc = get_task_description(task_id, num_modalities, modality_names)
        fig.suptitle(f'Task {task_id}: {task_desc}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def compute_generation_metrics(
    original: torch.Tensor,
    generated: torch.Tensor,
    task_mask: torch.Tensor,
    channels_per_modality: Union[int, List[int], torch.Tensor],
    metrics: List[str] = ['mse', 'psnr', 'mae']
) -> Dict[str, float]:
    mask = task_mask.float()
    
    # Expand to channel dimension first
    if isinstance(channels_per_modality, list):
        repeats = torch.tensor(channels_per_modality, device=mask.device)
        mask = mask.repeat_interleave(repeats)
    else:
        mask = mask.repeat_interleave(channels_per_modality)
    
    # Then expand to spatial dimensions
    for _ in range(original.dim() - 1):
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(original)
    
    # Compute only on generated modalities
    orig_masked = original * mask
    gen_masked = generated * mask
    
    results = {}
    
    if 'mse' in metrics:
        mse = ((orig_masked - gen_masked) ** 2).sum() / (mask.sum() + 1e-8)
        results['mse'] = mse.item()
    
    if 'mae' in metrics:
        mae = torch.abs(orig_masked - gen_masked).sum() / (mask.sum() + 1e-8)
        results['mae'] = mae.item()
    
    if 'psnr' in metrics and 'mse' in results:
        if results['mse'] > 0:
            psnr = 10 * np.log10(1.0 / results['mse'])
            results['psnr'] = psnr
        else:
            results['psnr'] = float('inf')
    
    return results


def evaluate_model(
    model: BaseGenerativeModel,
    dataloader: torch.utils.data.DataLoader,
    task_ids: List[int],
    num_samples: int = 100,
    num_steps: int = 50,
    device: str = 'cuda',
    **kwargs
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model on multiple tasks.
    
    Args:
        model: Trained model
        dataloader: Data loader
        task_ids: List of task IDs to evaluate
        num_samples: Number of samples to evaluate
        num_steps: Sampling steps
        device: Device to use
        **kwargs: Additional sampling parameters
        
    Returns:
        Dictionary mapping task_id to metrics
    """
    model.eval()
    results = {task_id: {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0} for task_id in task_ids}
    counts = {task_id: 0 for task_id in task_ids}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch['data'].shape[0] >= num_samples:
                break
            
            x_full = batch['data'].to(device)
            batch_size = x_full.shape[0]
            
            for task_id in task_ids:
                # Create task mask
                task_mask = task_to_binary_mask(task_id, model.num_modalities)
                task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(device)
                
                # Generate samples
                generated = model.sample(
                    x_condition=x_full,
                    task_mask=task_mask,
                    num_steps=num_steps,
                    **kwargs
                )
                
                # Compute metrics for each sample
                config = model.get_config()
                channels_per_modality = config['channels_per_modality']
                for i in range(batch_size):
                    metrics = compute_generation_metrics(
                        x_full[i],
                        generated[i],
                        task_mask[i],
                        channels_per_modality=channels_per_modality,
                    )
                    
                    for key, value in metrics.items():
                        results[task_id][key] += value
                    counts[task_id] += 1
    
    # Average metrics
    for task_id in task_ids:
        if counts[task_id] > 0:
            for key in results[task_id]:
                results[task_id][key] /= counts[task_id]
    
    return results
