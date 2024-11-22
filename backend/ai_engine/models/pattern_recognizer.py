import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np

class PatternRecognizer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Pattern detection
        pattern_scores = self.pattern_detector(attn_out)
        
        return {
            'pattern_scores': pattern_scores,
            'attention_weights': attn_weights,
            'hidden_states': hidden
        }