import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np

class RiskAssessor(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dims: List[int] = [128, 64, 32],
                 num_risk_factors: int = 5):
        super().__init__()
        
        # Risk factor layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Risk factor heads
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ) for _ in range(num_risk_factors)
        ])
        
        # Overall risk assessment
        self.risk_combiner = nn.Sequential(
            nn.Linear(num_risk_factors, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Individual risk factors
        risk_factors = torch.cat([
            head(shared_features) for head in self.risk_heads
        ], dim=1)
        
        # Overall risk
        overall_risk = self.risk_combiner(risk_factors)
        
        return {
            'risk_factors': risk_factors,
            'overall_risk': overall_risk,
            'feature_embedding': shared_features
        }