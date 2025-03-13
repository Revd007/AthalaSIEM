import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ThreatDetector(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        model_config = config.get('model_settings', {}).get('unified_threat_detector', {})
        
        # Get model parameters from config or use defaults
        self.input_dim = model_config.get('input_dim', 128)
        self.hidden_dim = model_config.get('hidden_dim', 64)
        self.num_classes = model_config.get('num_classes', 2)
        self.num_patterns = model_config.get('num_patterns', 10)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        
        # Initialize layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_dim // 2, self.num_classes)
        
        # Pattern detection head
        self.pattern_head = nn.Linear(self.hidden_dim // 2, self.num_patterns)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # Get predictions
        logits = self.classifier(features)
        patterns = self.pattern_head(features)
        
        # Apply activations
        probs = torch.softmax(logits, dim=-1)
        pattern_scores = torch.sigmoid(patterns)
        
        return {
            'logits': logits,
            'probabilities': probs,
            'patterns': pattern_scores
        }