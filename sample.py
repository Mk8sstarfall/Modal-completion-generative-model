"""
Sampling script for multi-modal completion models.

Example usage:
    # Sample from a single task with synthetic data
    python sample.py --checkpoint outputs/checkpoints/final_checkpoint.pt \
        --mode single --task_id 5 --dataset_type synthetic
    
    # Sample from all tasks
    python sample.py --checkpoint outputs/checkpoints/final_checkpoint.pt \
        --mode all --dataset_type synthetic
    
    # Evaluate model with BraTS data
    python sample.py --checkpoint outputs/checkpoints/final_checkpoint.pt \
        --mode evaluate --dataset_type brats --data_root /path/to/brats
"""

import argparse
import inspect
import torch
import json
from pathlib import Path
from typing import Dict, Any, Type
from torch.utils.data import Dataset
from diffusers import UNet2DModel

from models import FlowMatchingModel, DDPMModel
import data
from utils import (
    sample_conditional,
    sample_multiple_tasks,
    visualize_samples,
    evaluate_model,
    task_to_binary_mask,
    get_task_description,
)


def get_available_datasets() -> Dict[str, Type[Dataset]]:
    """
    Automatically discover all available dataset classes from data module.
    
    Returns:
        Dictionary mapping dataset name to dataset class
    """
    datasets = {}
    
    # Get all classes from data module's __all__
    if hasattr(data, '__all__'):
        for name in data.__all__:
            cls = getattr(data, name)
            # Check if it's a class and ends with 'Dataset'
            if inspect.isclass(cls) and name.endswith('Dataset'):
                # Convert class name to dataset type name
                # e.g., BraTSDataset -> brats, SyntheticModalDataset -> syntheticmodal
                dataset_type = name.replace('Dataset', '').lower()
                datasets[dataset_type] = cls
    
    return datasets


def get_class_init_params(cls: Type) -> set:
    """
    Get the parameter names of a class's __init__ method.
    
    Args:
        cls: Class to inspect
        
    Returns:
        Set of parameter names (excluding 'self')
    """
    sig = inspect.signature(cls.__init__)
    return {param for param in sig.parameters.keys() if param != 'self'}


def filter_kwargs_for_class(cls: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only include parameters accepted by the class's __init__.
    
    Args:
        cls: Class to filter kwargs for
        kwargs: Dictionary of keyword arguments
        
    Returns:
        Filtered dictionary with only valid parameters
    """
    valid_params = get_class_init_params(cls)
    return {k: v for k, v in kwargs.items() if k in valid_params}


def create_unet_backbone(
    in_channels: int,
    out_channels: int,
    block_out_channels: tuple = (64, 128, 256, 512),
    attention_head_dim: int = 8,
    num_modalities: int = 4,
) -> UNet2DModel:
    """
    Create a U-Net backbone from diffusers.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        block_out_channels: Channel dimensions for each block
        attention_head_dim: Attention head dimension
        num_modalities: Number of modalities (for class embedding)
        
    Returns:
        UNet2DModel instance
    """
    model = UNet2DModel(
        sample_size=64,  # Will be overridden by actual input size
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=attention_head_dim,
        class_embed_type='timestep',
        num_class_embeds=(1 << num_modalities),
    )
    return model


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint with improved config handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model and config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    
    # Determine model type from config
    if 'sigma_min' in config:
        model_class = FlowMatchingModel
        print("Detected Flow Matching model")
    else:
        model_class = DDPMModel
        print("Detected DDPM model")
    
    # Extract UNet configuration from config
    if isinstance(config['channels_per_modality'], int):
        total_channels = config['num_modalities'] * config['channels_per_modality']
    else:
        total_channels = sum(config['channels_per_modality'])
    
    # Get UNet channels from config or use defaults
    unet_channels = config.get('unet_channels', [64, 128, 256, 512])
    attention_head_dim = config.get('attention_head_dim', 8)
    
    # Create backbone with config from checkpoint
    backbone = create_unet_backbone(
        in_channels=total_channels,
        out_channels=total_channels,
        block_out_channels=tuple(unet_channels),
        attention_head_dim=attention_head_dim,
        num_modalities=config['num_modalities'],
    )
    
    # Create model using from_config
    model = model_class.from_config(config, backbone)
    
    # Load weights (try both regular and EMA)
    if 'ema_model_state_dict' in checkpoint:
        print("Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("Loading regular model weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model, config


def create_dataset(args, config, split='test'):
    """
    Create dataset based on arguments using dynamic class loading.
    
    Args:
        args: Command line arguments
        config: Model config from checkpoint
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        Dataset instance
    """
    # Get available datasets
    available_datasets = get_available_datasets()
    
    # Normalize dataset type name
    dataset_type = args.dataset_type.lower()
    
    # Check if dataset type is available
    if dataset_type not in available_datasets:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available options: {', '.join(sorted(available_datasets.keys()))}"
        )
    
    # Get the dataset class
    dataset_cls = available_datasets[dataset_type]
    
    # Prepare dataset parameters
    args_dict = vars(args)
    dataset_kwargs = {'split': split}
    
    # Add parameters from args
    for key, value in args_dict.items():
        dataset_kwargs[key] = value
    
    # Also add parameters from model config if needed
    dataset_kwargs['num_modalities'] = config.get('num_modalities', args.num_modalities)
    dataset_kwargs['channels_per_modality'] = config.get('channels_per_modality', args.channels_per_modality)
    
    # Filter kwargs to only include valid parameters for this dataset class
    filtered_kwargs = filter_kwargs_for_class(dataset_cls, dataset_kwargs)
    
    # Create dataset
    try:
        dataset = dataset_cls(**filtered_kwargs)
    except TypeError as e:
        # Provide helpful error message
        valid_params = get_class_init_params(dataset_cls)
        provided_params = set(filtered_kwargs.keys())
        print(f"Error creating {dataset_cls.__name__}:")
        print(f"  Valid parameters: {sorted(valid_params)}")
        print(f"  Provided parameters: {sorted(provided_params)}")
        raise
    
    return dataset


def sample_single_task(args):
    """
    Sample from a single task.
    """
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Create dataset for test data
    print(f"Creating {args.dataset_type} dataset...")
    dataset = create_dataset(args, config, split='test')

    modality_info = dataset.get_modality_info()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample_idx = args.sample_idx
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} out of range (dataset size: {len(dataset)})")
    
    data_dict = dataset[sample_idx]
    x_full = data_dict['data'].unsqueeze(0).to(args.device)
    
    # Create task mask
    task_mask = task_to_binary_mask(args.task_id, config['num_modalities'])
    task_mask = task_mask.unsqueeze(0).to(args.device)
    
    # Print task description
    task_desc = get_task_description(
        args.task_id,
        config['num_modalities'],
        modality_info['names'],
    )
    print(f"\nTask {args.task_id}: {task_desc}")
    
    # Generate sample
    print(f"Generating with {args.num_steps} steps...")
    with torch.no_grad():
        generated = sample_conditional(
            model=model,
            x_condition=x_full,
            task_mask=task_mask,
            num_steps=args.num_steps,
            device=args.device,
        )
    
    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f'sample_{sample_idx}_task_{args.task_id}.png'
    
    fig = visualize_samples(
        original=x_full[0],
        generated=generated[0],
        task_mask=task_mask[0],
        num_modalities=config['num_modalities'],
        channels_per_modality=config['channels_per_modality'],
        save_path=save_path,
        modality_names=modality_info['names'],
        task_id=args.task_id,
    )
    
    print(f"Visualization saved to {save_path}")
    
    # Save generated sample
    torch.save({
        'original': x_full.cpu(),
        'generated': generated.cpu(),
        'task_mask': task_mask.cpu(),
        'task_id': args.task_id,
        'config': config,
    }, output_dir / f'sample_{sample_idx}_task_{args.task_id}.pt')
    
    print(f"Sample data saved to {output_dir / f'sample_{sample_idx}_task_{args.task_id}.pt'}")


def sample_all_tasks(args):
    """
    Sample from all possible tasks for comparison.
    """
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Create dataset
    print(f"Creating {args.dataset_type} dataset...")
    dataset = create_dataset(args, config, split='test')

    modality_info = dataset.get_modality_info()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample_idx = args.sample_idx
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} out of range (dataset size: {len(dataset)})")
    
    data_dict = dataset[sample_idx]
    x_full = data_dict['data'].unsqueeze(0).to(args.device)
    
    # Generate for all non-trivial tasks
    max_task = 2 ** config['num_modalities'] - 1
    task_ids = list(range(1, max_task))  # Exclude 0 (all conditions) and 2^n-1 (all generate)
    
    print(f"Generating samples for {len(task_ids)} tasks...")
    with torch.no_grad():
        results = sample_multiple_tasks(
            model=model,
            x_full=x_full,
            task_ids=task_ids,
            num_steps=args.num_steps,
            device=args.device,
        )
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task_id, generated in results.items():
        task_mask = task_to_binary_mask(task_id, config['num_modalities'])
        save_path = output_dir / f'sample_{sample_idx}_task_{task_id}.png'
        
        visualize_samples(
            original=x_full[0],
            generated=generated[0],
            task_mask=task_mask,
            num_modalities=config['num_modalities'],
            channels_per_modality=config['channels_per_modality'],
            save_path=save_path,
            modality_names=modality_info['names'],
            task_id=task_id,
        )
    
    print(f"All visualizations saved to {output_dir}")


def evaluate_model_performance(args):
    """
    Evaluate model on test/validation set.
    """
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Create dataset
    print(f"Creating {args.dataset_type} dataset...")
    dataset = create_dataset(args, config, split='test')

    modality_info = dataset.get_modality_info()
    
    print(f"Dataset size: {len(dataset)}")
    
    # Limit dataset size if specified
    if args.num_eval_samples < len(dataset):
        indices = list(range(args.num_eval_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using subset of {args.num_eval_samples} samples")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Define tasks to evaluate
    max_task = 2 ** config['num_modalities'] - 1
    task_ids = list(range(1, min(max_task, args.max_eval_tasks + 1)))
    
    print(f"Evaluating on {len(task_ids)} tasks with {len(dataset)} samples...")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        task_ids=task_ids,
        num_samples=len(dataset),
        num_steps=args.num_steps,
        device=args.device,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 80)
    for task_id in task_ids:
        task_desc = get_task_description(task_id, config['num_modalities'], modality_info['names'])
        metrics = results[task_id]
        print(f"\nTask {task_id}: {task_desc}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("=" * 80)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_serializable = {
        str(k): v for k, v in results.items()
    }
    
    # Add config info
    results_serializable['config'] = {
        'num_modalities': config['num_modalities'],
        'channels_per_modality': config['channels_per_modality'],
        'dataset_type': args.dataset_type,
        'num_samples': len(dataset),
        'num_steps': args.num_steps,
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


def parse_channel_value(value):
    if isinstance(value, int):
        return value
    try:
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        values = [int(x.strip()) for x in value.split(',')]
        return values if len(values) > 1 else values[0]
    except:
        raise


def setup_parser():
    """Setup argument parser with dynamically discovered datasets."""
    parser = argparse.ArgumentParser(
        description='Sample from trained multi-modal completion model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Get available datasets
    available_datasets = get_available_datasets()
    dataset_choices = sorted(available_datasets.keys())
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint'
    )
    
    # Sampling mode
    mode_group = parser.add_argument_group('Sampling Mode')
    mode_group.add_argument(
        '--mode', 
        type=str, 
        default='single',
        choices=['single', 'all', 'evaluate'],
        help='Sampling mode: single task, all tasks, or evaluate'
    )
    
    # Task arguments
    task_group = parser.add_argument_group('Task')
    task_group.add_argument(
        '--task_id', 
        type=int, 
        default=1,
        help='Task ID for single task sampling (binary encoding)'
    )
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument(
        '--dataset_type', 
        type=str, 
        default='syntheticmodal',
        choices=dataset_choices if dataset_choices else None,
        help=f'Type of dataset to use. Available: {", ".join(dataset_choices)}'
    )
    dataset_group.add_argument(
        '--data_root', 
        type=str, 
        default=None,
        help='Root directory of the dataset (for real datasets like BraTS)'
    )
    dataset_group.add_argument(
        '--num_samples', 
        type=int, 
        default=100,
        help='Number of samples in synthetic dataset'
    )
    dataset_group.add_argument(
        '--sample_idx', 
        type=int, 
        default=0,
        help='Index of sample to use for visualization'
    )
    dataset_group.add_argument(
        '--image_size', 
        type=int, 
        default=64,
        help='Image size for synthetic dataset'
    )
    
    # Model configuration (override checkpoint config if needed)
    config_group = parser.add_argument_group('Model Configuration Override')
    config_group.add_argument(
        '--num_modalities', 
        type=int, 
        default=None,
        help='Number of modalities (usually inferred from checkpoint)'
    )
    config_group.add_argument(
        '--channels_per_modality', 
        type=parse_channel_value, 
        default=None,
        help='Channels per modality (usually inferred from checkpoint)'
    )
    
    # Sampling arguments
    sampling_group = parser.add_argument_group('Sampling')
    sampling_group.add_argument(
        '--num_steps', 
        type=int, 
        default=50,
        help='Number of sampling/denoising steps'
    )
    sampling_group.add_argument(
        '--batch_size', 
        type=int, 
        default=16,
        help='Batch size for evaluation'
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument(
        '--num_eval_samples', 
        type=int, 
        default=100,
        help='Number of samples to use for evaluation'
    )
    eval_group.add_argument(
        '--max_eval_tasks', 
        type=int, 
        default=15,
        help='Maximum number of tasks to evaluate'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples',
        help='Output directory for samples and results'
    )
    
    # Other
    other_group = parser.add_argument_group('Other')
    other_group.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='Device to use for inference'
    )
    other_group.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = setup_parser()
    args = parser.parse_args()
    return args


def main(args):
    """Main sampling function."""
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Run appropriate mode
    if args.mode == 'single':
        sample_single_task(args)
    elif args.mode == 'all':
        sample_all_tasks(args)
    elif args.mode == 'evaluate':
        evaluate_model_performance(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    args = parse_args()
    main(args)