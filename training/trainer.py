"""
Trainer class for multi-modal completion models.

Handles training loop, checkpointing, logging, and evaluation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from models.base_model import BaseGenerativeModel
from utils.modal_utils import sample_random_task, task_to_binary_mask


class Trainer:
    """
    Trainer for multi-modal completion models.
    """
    
    def __init__(self,
                 model: BaseGenerativeModel,
                 optimizer: Optimizer,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 output_dir: str = './outputs',
                 log_interval: int = 100,
                 save_interval: int = 1000,
                 eval_interval: int = 1000,
                 max_grad_norm: Optional[float] = 1.0,
                 ema_decay: Optional[float] = None,
                 **kwargs):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            scheduler: Optional learning rate scheduler
            device: Device to train on
            output_dir: Directory for saving checkpoints and logs
            log_interval: Steps between logging
            save_interval: Steps between saving checkpoints
            eval_interval: Steps between validation
            max_grad_norm: Maximum gradient norm for clipping
            ema_decay: Optional EMA decay rate for model parameters
            **kwargs: Additional trainer parameters
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        
        # EMA model
        self.ema_decay = ema_decay
        self.ema_model = None
        if ema_decay is not None and ema_decay > 0:
            self.ema_model = self._create_ema_model()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def _create_ema_model(self) -> nn.Module:
        """Create EMA version of the model."""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        return ema_model
    
    def _update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move data to device
        x_0 = batch['data'].to(self.device)
        
        # Sample random task for each sample in batch
        batch_size = x_0.shape[0]
        device = x_0.device
        task_id = sample_random_task(
            self.model.num_modalities,
            exclude_empty=False
        )
        task_mask = task_to_binary_mask(task_id, self.model.num_modalities)
        task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Compute loss
        loss_dict = self.model.compute_loss(x_0, task_mask=task_mask)
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Convert tensors to floats for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in loss_dict.items()}
        
        return metrics
    
    @torch.no_grad()
    def eval_step(self) -> Dict[str, float]:
        """
        Perform validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            x_0 = batch['data'].to(self.device)
            
            # Use a fixed set of tasks for validation
            batch_size = x_0.shape[0]
            # Evaluate on a few representative tasks
            for task_id in range(min(4, 2 ** self.model.num_modalities)):
                task_mask = task_to_binary_mask(task_id, self.model.num_modalities)
                task_mask = task_mask.unsqueeze(0).expand(batch_size, -1).to(self.device)
                
                loss_dict = self.model.compute_loss(x_0, task_id=task_id, task_mask=task_mask)
                total_loss += loss_dict['loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = []
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_metrics = {
                        k: sum(m[k] for m in epoch_metrics[-self.log_interval:]) / 
                           min(self.log_interval, len(epoch_metrics))
                        for k in metrics.keys()
                    }
                    
                    log_dict = {
                        'epoch': epoch,
                        'step': self.global_step,
                        **avg_metrics,
                    }
                    
                    if self.scheduler is not None:
                        log_dict['lr'] = self.scheduler.get_last_lr()[0]
                    
                    print(f"\nStep {self.global_step}: " + 
                          ", ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
                
                # Validation
                if self.global_step % self.eval_interval == 0 and self.val_dataloader is not None:
                    val_metrics = self.eval_step()
                    print(f"\nValidation: " + 
                          ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            # Save checkpoint at end of epoch
            # self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        print("Training completed!")
        
        # Save final checkpoint
        self.save_checkpoint('final_checkpoint.pt')
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_config(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Also save model config as JSON
        config_path = self.checkpoint_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.model.get_config(), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")
