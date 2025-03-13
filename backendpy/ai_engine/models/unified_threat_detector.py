import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Any, Optional, Union, List, Tuple
from .base_model import BaseModel
import logging
import numpy as np

class UnifiedThreatDetector(BaseModel):
    """
    Unified Threat Detector yang menggabungkan analisis teks dan numerik
    dengan multiple detection heads dan robust error handling
    """
    def __init__(self, config: Dict[str, Any]):
        # Validasi config sebelum inisialisasi
        self._validate_config(config)
        
        # Initialize parent
        super().__init__(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        try:
            # Core parameters
            self.input_dim = 406  # Set to match jumlah fitur dari dataset
            self.hidden_dim = config.get('hidden_dim', 256)
            self.num_classes = config.get('num_classes', 2)
            self.dropout_rate = config.get('dropout_rate', 0.3)  # Increased dropout
            
            # Build model components with stronger regularization
            self.numeric_processor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim),  # Added BatchNorm
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim // 2),  # Added BatchNorm
                nn.Dropout(self.dropout_rate)
            )
            
            # Classification head with L2 regularization
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim // 4),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 4, self.num_classes)
            )
            
            # Initialize weights with smaller values
            self.apply(self._init_weights)
            
            # Move to device
            self.device = torch.device(config.get('device', 'cpu'))
            self.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing UnifiedThreatDetector: {e}")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required_fields = ['input_dim', 'hidden_dim']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Set default values for optional fields
        defaults = {
            'num_classes': 2,
            'num_patterns': 10,
            'num_behaviors': 5,
            'use_bert': True,
            'bert_model': 'bert-base-uncased',
            'dropout_rate': 0.2,
            'device': None
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value

    def _build_numeric_processor(self) -> nn.Module:
        """Build numeric feature processing layers"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout_rate', 0.2)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )

    def _process_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Process text input through BERT"""
        try:
            if not self.use_bert:
                return torch.zeros(1, self.bert_dim, device=self.device)
                
            # Handle single string or list of strings
            if isinstance(text, str):
                text = [text]
                
            # Tokenize and process through BERT
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                bert_output = self.bert(**tokens)
                
            return bert_output.last_hidden_state.mean(1)
            
        except Exception as e:
            self.logger.error(f"Error in text processing: {e}")
            return torch.zeros(len(text) if isinstance(text, list) else 1,
                             self.bert_dim, device=self.device)

    def _process_numeric(self, numeric: torch.Tensor) -> torch.Tensor:
        """Process numeric features"""
        try:
            if not isinstance(numeric, torch.Tensor):
                numeric = torch.tensor(numeric, dtype=torch.float32)
            numeric = numeric.to(self.device)
            
            # Handle different input dimensions
            if numeric.dim() == 1:
                numeric = numeric.unsqueeze(0)
            
            return self.numeric_processor(numeric)
            
        except Exception as e:
            self.logger.error(f"Error in numeric processing: {e}")
            return torch.zeros(1, self.hidden_dim, device=self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        try:
            # Ensure input is float tensor
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(self.device)
            
            # Process features
            features = self.numeric_processor(inputs)
            
            # Get classification output
            logits = self.classifier(features)
            
            return logits
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            # Return zero tensor with correct shape [batch_size, num_classes]
            return torch.zeros((inputs.size(0), self.num_classes), device=self.device)

    def _get_zero_outputs(self) -> Dict[str, torch.Tensor]:
        """Generate zero outputs for error cases"""
        return {
            'threat_score': torch.zeros(1, dtype=torch.float32, device=self.device),
            'patterns': torch.zeros(self.num_patterns, dtype=torch.float32, device=self.device),
            'behaviors': torch.zeros(self.num_behaviors, dtype=torch.float32, device=self.device)
        }

    def predict(self, 
                text_input: Union[str, List[str]], 
                numeric_input: torch.Tensor,
                threshold: float = None) -> Dict[str, Any]:
        """
        High-level prediction method with threshold application
        """
        with torch.no_grad():
            outputs = self.forward(text_input, numeric_input)
            
            threshold = threshold or self.config.get('threat_threshold', 0.7)
            
            return {
                'is_threat': outputs['threat_score'].item() > threshold,
                'threat_score': outputs['threat_score'].item(),
                'detected_patterns': outputs['patterns'].tolist(),
                'behavioral_indicators': outputs['behaviors'].tolist()
            }

    def _init_weights(self, module):
        """Initialize weights with smaller values to prevent overfitting"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)