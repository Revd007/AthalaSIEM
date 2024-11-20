import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import wandb  # untuk tracking eksperimen
from tqdm import tqdm

class TrainingManager:
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 experiment_name: str):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize training components
        self.criterion = self._initialize_criterion()
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'f1_score': []
        }
        
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb"""
        wandb.init(
            project="ai_model_training",
            name=self.experiment_name,
            config=self.config
        )
    
    async def train(self, 
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int):
        """Main training loop"""
        best_val_loss = float('inf')
        patience = self.config.get('patience', 5)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            try:
                # Training phase
                train_metrics = await self._train_epoch(train_loader, epoch)
                
                # Validation phase
                val_metrics = await self._validate_epoch(val_loader, epoch)
                
                # Update learning rate
                self.scheduler.step(val_metrics['val_loss'])
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Save checkpoint if best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, best_val_loss)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
                
            except Exception as e:
                self.logger.error(f"Error in epoch {epoch}: {e}")
                raise
    
    async def _train_epoch(self, 
                          train_loader: DataLoader,
                          epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch['labels'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        return {
            'train_loss': total_loss / len(train_loader),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    async def _validate_epoch(self, 
                            val_loader: DataLoader,
                            epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
        
        return {'val_loss': total_loss / len(val_loader)}
    
    def _log_metrics(self, 
                    epoch: int,
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float]):
        """Log metrics to wandb and local storage"""
        # Update local metrics
        self.metrics['train_loss'].append(train_metrics['train_loss'])
        self.metrics['val_loss'].append(val_metrics['val_loss'])
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            **train_metrics,
            **val_metrics
        })
        
        # Save metrics to file
        self._save_metrics()
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config['checkpoint_dir'])
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'metrics': self.metrics
        }
        
        torch.save(
            checkpoint,
            checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
        )