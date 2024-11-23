import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class ThreatDetector(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Get model parameters from config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Define model layers
        self.layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) 
            for _ in range(self.num_layers)
        ])
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_layer = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout_layer(x)
        return torch.sigmoid(self.output_layer(x))