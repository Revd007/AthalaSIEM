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

class TrainingManager:
    def __init__(self, config: Dict[str, Any], experiment_name: str = "default_experiment"):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training metrics
        self.training_history = []
        self.best_metrics = {}
        
        self.model = None  # Akan diset saat training dimulai
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb"""
        wandb.init(
            project="ai_model_training",
            name=self.config['experiment_name'],
            config=self.config
        )
    
    async def train(self, train_loader, val_loader, num_epochs: int) -> Dict[str, Any]:
        """Train models using provided data loaders"""
        try:
            # Get default model from model manager
            self.model = self.model_manager.get()  # Tanpa parameter tambahan
            if not self.model:
                raise ValueError("No model available for training")

            training_results = {}
            
            # Get models from model manager passed through train parameters
            for epoch in range(num_epochs):
                epoch_metrics = {
                    'train_loss': [],
                    'val_loss': [],
                    'accuracy': []
                }
                
                # Training phase
                self.model.train()
                train_metrics = await self._train_epoch(train_loader, epoch)
                epoch_metrics['train_loss'].append(train_metrics['loss'])

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    val_metrics = await self._validate_epoch(val_loader)
                    epoch_metrics['val_loss'].append(val_metrics['loss'])
                    epoch_metrics['accuracy'].append(val_metrics['accuracy'])

                # Log metrics
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'accuracy': val_metrics['accuracy']
                })

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_metrics['loss'])

                # Save best model
                if not self.best_metrics or val_metrics['loss'] < self.best_metrics['val_loss']:
                    self.best_metrics = val_metrics
                    self.save_model(f'best_model_epoch_{epoch}.pt')

                # Early stopping check
                if self._should_stop_early(val_metrics['loss']):
                    self.logger.info("Early stopping triggered")
                    break
                # ...
                
                training_results[f'epoch_{epoch}'] = epoch_metrics
                
            return {
                'status': 'success',
                'metrics': training_results,
                'final_loss': epoch_metrics['val_loss'][-1]
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return {'error': str(e)}
    
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