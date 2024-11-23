import torch
import torch.nn as nn
from typing import *

class ThreatDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Fix tensor creation by specifying device properly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(self.device)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4,
            dropout=0.1
        ).to(self.device)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure input tensor is on correct device
        x = x.to(self.device)
        
        # Create properly sized empty tensor for attention mask
        attention_mask = torch.empty(
            (x.size(0), x.size(1)),
            dtype=torch.float32,
            device=self.device
        )
        
        # Process input through encoder
        encoded = self.encoder(x)
        
        # Apply attention mechanism
        attended, attention_weights = self.attention(
            encoded, encoded, encoded,
            attn_mask=attention_mask
        )
        
        return {
            'output': attended,
            'attention_weights': attention_weights
        }

    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.forward(x)
            attention_maps = outputs['attention_weights']
            
        return attention_maps