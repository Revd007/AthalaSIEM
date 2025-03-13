import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import wandb
import gc

from ai_engine.core.model_manager import ModelManager

class TrainingManager:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        model_manager: ModelManager,
        experiment_name: str
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.model_manager = model_manager
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training metrics
        self.training_history = []
        self.best_metrics = {}
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb"""
        if self.config.get('enable_wandb', False):
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                name=self.config['wandb']['experiment_name'],
                tags=self.config['wandb']['tags'],
                config=self.config['wandb']['config']
            )
        else:
            self.logger.info("Wandb tracking is disabled in configuration")
    
    async def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        learning_rate: float
    ):
        try:
            # Initialize optimizer with weight decay (L2 regularization)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01  # Added L2 regularization
            )
            
            # Initialize learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = optimizer

            self.model.to(self.device)
            self.model.train()

            total_steps = len(train_loader)
            best_loss = float('inf')
            patience = self.config.get('training', {}).get('early_stopping_patience', 5)
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    inputs = batch['features'].float().to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()
                    
                    try:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Add L2 regularization loss
                        l2_lambda = 0.01
                        l2_reg = torch.tensor(0.).to(self.device)
                        for param in self.model.parameters():
                            l2_reg += torch.norm(param)
                        loss += l2_lambda * l2_reg
                        
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        if batch_idx % 100 == 0:
                            self.logger.info(f'Epoch [{epoch+1}/{epochs}] '
                                           f'Step [{batch_idx}/{total_steps}] '
                                           f'Loss: {loss.item():.4f}')
                            
                    except Exception as e:
                        self.logger.error(f"Forward/loss error: {str(e)}")
                        continue

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        inputs = batch['features'].float().to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_batches += 1

                avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                
                self.logger.info(f'Epoch [{epoch+1}/{epochs}] '
                               f'Train Loss: {avg_train_loss:.4f} '
                               f'Val Loss: {avg_val_loss:.4f}')

                # Learning rate scheduling
                scheduler.step(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch, avg_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            return {
                "status": "Training completed",
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "best_val_loss": best_loss
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
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
                
                # Add memory management
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Add checkpointing
            if epoch % self.config['save_interval'] == 0:
                checkpoint_path = f"checkpoints/epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
        
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
        try:
            # Get checkpoint directory from config or use default
            checkpoint_dir = Path(self.config.get('training', {}).get('checkpoint_dir', 'checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create checkpoint filename
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}_loss_{val_loss:.4f}.pt'
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_loss,
                'config': self.config
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Keep only the last N checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir, keep_last=5)
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_last: int = 5):
        """Clean up old checkpoints, keeping only the last N"""
        try:
            checkpoints = sorted(checkpoint_dir.glob('model_epoch_*.pt'))
            if len(checkpoints) > keep_last:
                for checkpoint in checkpoints[:-keep_last]:
                    checkpoint.unlink()
                    self.logger.debug(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")
    
    def _initialize_criterion(self) -> nn.Module:
        """Initialize the loss criterion"""
        criterion_name = self.config.get('criterion', 'CrossEntropyLoss')
        if criterion_name == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss()
        elif criterion_name == 'MSELoss':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer"""
        optimizer_name = self.config.get('optimizer', 'Adam')
        lr = self.config.get('learning_rate', 1e-4)
        
        if optimizer_name == 'Adam':
            return torch.optim.Adam(
                self.models['anomaly_detector'].parameters(),
                lr=lr,
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_name == 'AdamW':
            return torch.optim.AdamW(
                self.models['anomaly_detector'].parameters(),
                lr=lr,
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(
                self.models['anomaly_detector'].parameters(),
                lr=lr,
                momentum=self.config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _initialize_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Initialize the learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('lr_factor', 0.1),
                patience=self.config.get('lr_patience', 3),
                verbose=True
            )
        elif scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('t_max', 10),
                eta_min=self.config.get('min_lr', 1e-6)
            )