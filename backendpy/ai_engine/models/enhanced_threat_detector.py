import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Any, Optional
from .base_model import BaseModel

class EnhancedThreatDetector(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize BERT dan tokenizer
        self.bert = BertModel.from_pretrained(config.get('bert_model', 'bert-base-uncased'))
        self.tokenizer = BertTokenizer.from_pretrained(config.get('bert_model', 'bert-base-uncased'))
        
        # Numeric features processing
        self.numeric_layers = nn.Sequential(
            nn.Linear(self.config['input_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        combined_dim = 768 + self.config['hidden_dim']  # BERT dim + hidden dim
        self.threat_head = nn.Linear(combined_dim, 1)
        self.pattern_head = nn.Linear(combined_dim, self.config.get('num_patterns', 10))
        self.behavior_head = nn.Linear(combined_dim, self.config.get('num_behaviors', 5))
        
        # Move to appropriate device
        self.to(self.device)
    
    def _get_zero_outputs(self):
        """Return zero tensors for error cases"""
        return {
            'threat_score': torch.zeros(1, dtype=torch.float32, device=self.device),
            'patterns': torch.zeros(self.config.get('num_patterns', 10), dtype=torch.float32, device=self.device),
            'behaviors': torch.zeros(self.config.get('num_behaviors', 5), dtype=torch.float32, device=self.device)
        }
    
    def preprocess_input(self, text_input: str, numeric_input: torch.Tensor):
        """Preprocess both text and numeric inputs"""
        # Tokenize text
        tokens = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Process numeric input
        if not isinstance(numeric_input, torch.Tensor):
            numeric_input = torch.tensor(numeric_input, dtype=torch.float32)
        numeric_input = numeric_input.to(self.device)
        
        return tokens, numeric_input

    def forward(self, text_input: str, numeric_input: torch.Tensor):
        try:
            # Preprocess inputs
            tokens, numeric_features = self.preprocess_input(text_input, numeric_input)
            
            # Get BERT features
            text_features = self.bert(**tokens).last_hidden_state.mean(1)
            
            # Process numeric features
            numeric_features = self.numeric_layers(numeric_features)
            
            # Combine features
            combined = torch.cat([text_features, numeric_features], dim=-1)
            
            # Get predictions
            return {
                'threat_score': torch.sigmoid(self.threat_head(combined)),
                'patterns': torch.softmax(self.pattern_head(combined), dim=-1),
                'behaviors': torch.softmax(self.behavior_head(combined), dim=-1)
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced threat detector: {e}")
            return self._get_zero_outputs()