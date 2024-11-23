import torch
import torch.nn as nn
from typing import Dict, Any

class ThreatDetector(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_classes = config.get('num_classes', 2)
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)