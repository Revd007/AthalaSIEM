import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import json
import os

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.version = config.get('version', '1.0')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to be implemented by child classes"""
        raise NotImplementedError
    
    def save_model(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model with metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state': self.state_dict(),
            'config': self.config,
            'version': self.version,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        
    @classmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """Load model with metadata"""
        save_dict = torch.load(path)
        
        model = cls(save_dict['config'])
        model.load_state_dict(save_dict['model_state'])
        model.version = save_dict['version']
        
        return model, save_dict['metadata']
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get model parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }