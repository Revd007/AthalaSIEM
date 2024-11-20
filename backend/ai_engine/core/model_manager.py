from typing import Dict, Any, Optional
import torch
import logging
from datetime import datetime
import os
import json
from ..models import BaseModel

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_model = None
        self.model_history = []
        
    async def train_model(self,
                         train_loader: torch.utils.data.DataLoader,
                         test_loader: torch.utils.data.DataLoader) -> BaseModel:
        """Train model"""
        model = self._create_model()
        optimizer = self._create_optimizer(model)
        criterion = self._create_criterion()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                features = batch['features']
                labels = batch['label']
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_model(model, test_loader, criterion)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self._save_model(model, {'val_loss': val_loss})
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
        
        self.current_model = model
        return model
    
    def _create_model(self) -> BaseModel:
        """Create model instance"""
        model_class = self.config['model_class']
        return model_class(self.config)
    
    def _create_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        """Create optimizer"""
        return torch.optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
    
    def _create_criterion(self) -> torch.nn.Module:
        """Create loss criterion"""
        return torch.nn.CrossEntropyLoss()
    
    def _validate_model(self,
                       model: BaseModel,
                       test_loader: torch.utils.data.DataLoader,
                       criterion: torch.nn.Module) -> float:
        """Validate model"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features']
                labels = batch['label']
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
        return total_loss / len(test_loader)
    
    def _save_model(self,
                   model: BaseModel,
                   metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"{self.config['checkpoint_dir']}/model_{timestamp}.pt"
        
        metadata = metadata or {}
        metadata.update({
            'timestamp': timestamp,
            'config': self.config
        })
        
        model.save_model(save_path, metadata)
        self.model_history.append(save_path)
        
        self.logger.info(f"Model saved: {save_path}")