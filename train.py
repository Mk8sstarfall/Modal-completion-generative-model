"""
Main training script for multi-modal completion models.

Example usage:
    # Train with synthetic data
    python train.py --model_type flow --num_modalities 4 --epochs 100 --dataset_type synthetic
    
    # Train with BraTS data
    python train.py --model_type flow --num_modalities 4 --epochs 100 \
        --dataset_type brats --data_root /path/to/brats --target_size 80 80 80
"""

import argparse
import inspect
import importlib
from typing import Dict, Any, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import UNet2DModel

from models import FlowMatchingModel, DDPMModel
import data
from training import Trainer


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


def create_model(args):
    """
    Create model based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Model instance
    """
    # Calculate total channels
    if isinstance(args.channels_per_modality, int):
        total_channels = args.num_modalities * args.channels_per_modality
    else:
        total_channels = sum(args.channels_per_modality)
    
    # Create backbone
    backbone = create_unet_backbone(
        in_channels=total_channels,
        out_channels=total_channels,
        block_out_channels=tuple(args.unet_channels),
        attention_head_dim=args.attention_head_dim,
        num_modalities=args.num_modalities,
    )
    
    # Create model based on type
    if args.model_type == 'flow':
        model = FlowMatchingModel(
            backbone=backbone,
            num_modalities=args.num_modalities,
            channels_per_modality=args.channels_per_modality,
            sigma_min=args.sigma_min,
            path_type=args.path_type,
        )
    elif args.model_type == 'ddpm':
        model = DDPMModel(
            backbone=backbone,
            num_modalities=args.num_modalities,
            channels_per_modality=args.channels_per_modality,
            num_train_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model


def create_dataset(args, split='train'):
    """
    Create dataset based on arguments using dynamic class loading.
    
    Args:
        args: Command line arguments
        split: Dataset split ('train' or 'val')
        
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
    
    # Prepare dataset parameters from args
    # Convert args namespace to dictionary
    args_dict = vars(args)
    
    # Add split parameter
    dataset_kwargs = {'split': split}
    
    # Special handling for different dataset parameters
    
    # Get all potential dataset parameters
    for key, value in args_dict.items():
        dataset_kwargs[key] = value
    
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


def main(args):
    """Main training function."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create datasets
    print(f"Creating {args.dataset_type} datasets...")
    train_dataset = create_dataset(args, split='train')
    val_dataset = create_dataset(args, split='val')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(args)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.learning_rate * 0.01,
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        max_grad_norm=args.max_grad_norm,
        ema_decay=args.ema_decay,
    )
    
    # Load checkpoint if resuming
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=args.epochs)


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
        description='Train multi-modal completion model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Get available datasets
    available_datasets = get_available_datasets()
    dataset_choices = sorted(available_datasets.keys())
    
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
        help='Root directory of the dataset'
    )
    
    # Common dataset parameters (may be used by multiple datasets)
    dataset_group.add_argument('--image_size', type=int, default=64,
                              help='Image size')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model_type', type=str, default='flow',
                            choices=['flow', 'ddpm'],
                            help='Type of generative model')
    model_group.add_argument('--num_modalities', type=int, default=4,
                            help='Number of modalities')
    model_group.add_argument('--channels_per_modality', type=parse_channel_value, default=1,
                            help='Number of channels per modality')
    model_group.add_argument('--unet_channels', nargs='+', type=int,
                            default=[64, 128, 256, 512],
                            help='U-Net block output channels')
    model_group.add_argument('--attention_head_dim', type=int, default=8,
                            help='Attention head dimension')
    
    # Flow Matching specific
    flow_group = parser.add_argument_group('Flow Matching')
    flow_group.add_argument('--sigma_min', type=float, default=1e-4,
                           help='Minimum noise level for Flow Matching')
    flow_group.add_argument('--path_type', type=str, default='linear',
                           choices=['linear', 'vp', 'vp_simple'],
                           help='Type of interpolation path')
    
    # DDPM specific
    ddpm_group = parser.add_argument_group('DDPM')
    ddpm_group.add_argument('--num_timesteps', type=int, default=1000,
                           help='Number of diffusion timesteps for DDPM')
    ddpm_group.add_argument('--beta_schedule', type=str, default='linear',
                           choices=['linear', 'cosine', 'scaled_linear'],
                           help='Beta schedule for DDPM')
    ddpm_group.add_argument('--prediction_type', type=str, default='epsilon',
                           choices=['epsilon', 'sample', 'v_prediction'],
                           help='Prediction type for DDPM')
    
    # Training arguments
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch_size', type=int, default=32,
                               help='Batch size')
    training_group.add_argument('--epochs', type=int, default=100,
                               help='Number of epochs')
    training_group.add_argument('--learning_rate', type=float, default=1e-4,
                               help='Learning rate')
    training_group.add_argument('--weight_decay', type=float, default=0.0,
                               help='Weight decay')
    training_group.add_argument('--beta1', type=float, default=0.9,
                               help='Adam beta1')
    training_group.add_argument('--beta2', type=float, default=0.999,
                               help='Adam beta2')
    training_group.add_argument('--max_grad_norm', type=float, default=1.0,
                               help='Maximum gradient norm for clipping')
    training_group.add_argument('--ema_decay', type=float, default=0.9999,
                               help='EMA decay rate (set to 0 to disable)')
    
    # Logging and checkpointing
    logging_group = parser.add_argument_group('Logging & Checkpointing')
    logging_group.add_argument('--output_dir', type=str, default='./outputs',
                              help='Output directory')
    logging_group.add_argument('--log_interval', type=int, default=100,
                              help='Steps between logging')
    logging_group.add_argument('--save_interval', type=int, default=5000,
                              help='Steps between saving checkpoints')
    logging_group.add_argument('--eval_interval', type=int, default=1000,
                              help='Steps between evaluation')
    
    # Other
    other_group = parser.add_argument_group('Other')
    other_group.add_argument('--device', type=str, default='cuda',
                            help='Device to use for training')
    other_group.add_argument('--num_workers', type=int, default=4,
                            help='Number of data loading workers')
    other_group.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    other_group.add_argument('--resume_from', type=str, default=None,
                            help='Path to checkpoint to resume from')
    
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = setup_parser()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)