import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

class PatternRecognizer(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        config = config or {}
        model_config = config.get('model_settings', {}).get('pattern_recognizer', {})
        
        # Get model parameters from config or use defaults
        try:
            self.input_dim = int(model_config.get('input_dim', 128))
            self.hidden_dim = int(model_config.get('hidden_dim', 64))
            self.num_layers = int(model_config.get('num_layers', 2))
            self.num_patterns = int(model_config.get('num_patterns', 10))
            self.dropout_rate = float(model_config.get('dropout_rate', 0.2))
            
            # Validate parameters
            if self.input_dim <= 0 or self.hidden_dim <= 0:
                raise ValueError("Input and hidden dimensions must be positive")
            if self.num_layers <= 0:
                raise ValueError("Number of layers must be positive")
            if not 0 <= self.dropout_rate <= 1:
                raise ValueError("Dropout rate must be between 0 and 1")
            
            # Initialize LSTM
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout_rate if self.num_layers > 1 else 0
            )
            
            # Initialize pattern detection layers
            self.pattern_detector = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_patterns)
            )
            
            self.logger.info("Successfully initialized PatternRecognizer with LSTM")
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error initializing PatternRecognizer: {e}")
            # Fallback to simpler model
            self.lstm = None
            self.input_dim = 128  # Default fallback values
            self.hidden_dim = 64
            self.num_patterns = 10
            self.dropout_rate = 0.2
            
            self.pattern_detector = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.num_patterns)
            )
            self.logger.warning("Initialized PatternRecognizer with fallback configuration")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            batch_size = x.size(0) if len(x.size()) > 1 else 1
            
            # Handle different input shapes
            if len(x.size()) == 2:  # [batch_size, features]
                x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, features]
            elif len(x.size()) == 1:  # Single sample
                x = x.view(1, 1, -1)  # [1, 1, features]
                
            # Ensure correct device
            device = next(self.parameters()).device
            x = x.to(device)
                
            # Process through LSTM if available
            if self.lstm is not None:
                lstm_out, _ = self.lstm(x)
                features = lstm_out[:, -1, :]  # Take last sequence output
            else:
                features = x.squeeze(1)  # Remove sequence dimension for linear layer
            
            # Detect patterns
            pattern_logits = self.pattern_detector(features)
            pattern_probs = torch.sigmoid(pattern_logits)
            
            return {
                'pattern_logits': pattern_logits,
                'pattern_probabilities': pattern_probs
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern recognition forward pass: {e}")
            # Return zero tensor with correct shape on error
            return {
                'pattern_logits': torch.zeros(batch_size, self.num_patterns, device=x.device),
                'pattern_probabilities': torch.zeros(batch_size, self.num_patterns, device=x.device)
            }

    def __str__(self):
        return (f"PatternRecognizer(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"num_patterns={self.num_patterns}, "
                f"using_lstm={self.lstm is not None})")