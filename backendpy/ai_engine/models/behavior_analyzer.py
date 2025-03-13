import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Any, Optional

class BehaviorAnalyzer(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        model_config = config.get('model_settings', {}).get('behavior_analyzer', {})
        
        # Get model parameters from config or use defaults
        self.input_dim = model_config.get('input_dim', 768)  # BERT default
        self.hidden_dim = model_config.get('hidden_dim', 256)
        self.num_behaviors = model_config.get('num_behaviors', 5)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        
        try:
            # Initialize BERT only if needed
            if model_config.get('use_bert', False):
                bert_model_name = model_config.get('bert_model', 'bert-base-uncased')
                self.bert = BertModel.from_pretrained(bert_model_name)
                self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            else:
                self.bert = None
                self.tokenizer = None
            
            # Behavior analysis layers
            self.behavior_classifier = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.num_behaviors)
            )
            
        except Exception as e:
            print(f"Error initializing BehaviorAnalyzer: {e}")
            # Fallback to simple model if BERT fails
            self.bert = None
            self.tokenizer = None
            self.behavior_classifier = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.num_behaviors)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            # Use BERT if available and input is text
            if self.bert is not None and isinstance(x, str):
                inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
                outputs = self.bert(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = x
            
            # Ensure correct input shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Get behavior predictions
            behavior_logits = self.behavior_classifier(features)
            behavior_probs = torch.sigmoid(behavior_logits)
            
            return {
                'behavior_logits': behavior_logits,
                'behavior_probabilities': behavior_probs
            }
            
        except Exception as e:
            print(f"Error in behavior analysis forward pass: {e}")
            # Return zero tensor with correct shape on error
            return {
                'behavior_logits': torch.zeros(1, self.num_behaviors),
                'behavior_probabilities': torch.zeros(1, self.num_behaviors)
            }