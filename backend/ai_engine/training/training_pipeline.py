from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from pathlib import Path
import json
from ..core.dataset_handler import UniversalDataset
from ..evaluation.metrics import AccuracyMetrics
from ..models.base_model import BaseModel

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = AccuracyMetrics()
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    async def train_model(
        self,
        model: BaseModel,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """Train an AI model"""
        try:
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(epochs):
                # Training phase
                train_metrics = await self._train_epoch(
                    model, train_data, optimizer
                )
                
                # Validation phase
                val_metrics = await self._validate_epoch(
                    model, val_data
                ) if val_data else None
                
                # Update learning rate
                scheduler.step(val_metrics['loss'] if val_metrics else train_metrics['loss'])
                
                # Save checkpoint if best model
                if val_metrics and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self._save_checkpoint(model, optimizer, epoch, val_metrics)
                
                # Record history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
                
            return {
                'training_history': training_history,
                'final_metrics': val_metrics or train_metrics,
                'model_path': str(self.checkpoint_dir / 'best_model.pt')
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise