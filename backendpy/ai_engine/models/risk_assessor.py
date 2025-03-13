import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

class RiskAssessor(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        config = config or {}
        model_config = config.get('model_settings', {}).get('risk_assessor', {})
        
        try:
            # Get model parameters from config or use defaults
            self.input_dim = int(model_config.get('input_dim', 256))
            self.hidden_dims = [
                int(dim) for dim in model_config.get('hidden_dims', [128, 64])
            ]
            self.output_dim = int(model_config.get('output_dim', 1))
            self.dropout_rate = float(model_config.get('dropout_rate', 0.2))
            
            # Validate parameters
            if self.input_dim <= 0 or any(dim <= 0 for dim in self.hidden_dims):
                raise ValueError("All dimensions must be positive")
            if not 0 <= self.dropout_rate <= 1:
                raise ValueError("Dropout rate must be between 0 and 1")
            
            # Build layers
            layers = []
            prev_dim = self.input_dim
            
            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ])
                prev_dim = hidden_dim
            
            # Final output layer
            layers.append(nn.Linear(prev_dim, self.output_dim))
            layers.append(nn.Sigmoid())  # For risk score between 0 and 1
            
            self.risk_assessor = nn.Sequential(*layers)
            self.logger.info("Successfully initialized RiskAssessor")
            
        except Exception as e:
            self.logger.error(f"Error initializing RiskAssessor: {e}")
            # Fallback to simple model
            self.input_dim = 256
            self.output_dim = 1
            self.risk_assessor = nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.output_dim),
                nn.Sigmoid()
            )
            self.logger.warning("Initialized RiskAssessor with fallback configuration")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            # Ensure input has correct shape
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Ensure correct device
            device = next(self.parameters()).device
            x = x.to(device)
            
            # Calculate risk score
            risk_score = self.risk_assessor(x)
            
            return {
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment forward pass: {e}")
            batch_size = x.size(0) if len(x.size()) > 1 else 1
            return {
                'risk_score': torch.zeros(batch_size, 1, device=x.device),
                'risk_level': torch.zeros(batch_size, 1, device=x.device)
            }

    def _get_risk_level(self, risk_score: torch.Tensor) -> torch.Tensor:
        """Convert risk score to categorical risk level"""
        risk_levels = torch.zeros_like(risk_score)
        risk_levels[risk_score > 0.7] = 3  # High risk
        risk_levels[torch.logical_and(risk_score > 0.3, risk_score <= 0.7)] = 2  # Medium risk
        risk_levels[risk_score <= 0.3] = 1  # Low risk
        return risk_levels

    def __str__(self):
        return (f"RiskAssessor(input_dim={self.input_dim}, "
                f"hidden_dims={getattr(self, 'hidden_dims', 'N/A')}, "
                f"output_dim={self.output_dim})")