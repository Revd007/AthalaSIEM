import torch
import torch.nn as nn
from transformers import BertModel
from typing import Dict, Any, List

class BehaviorAnalyzer(nn.Module):
    def __init__(self, 
                 bert_model: str = 'bert-base-uncased',
                 num_behaviors: int = 10,
                 hidden_dim: int = 768):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        
        self.behavior_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_behaviors),
            nn.Softmax(dim=-1)
        )
        
        self.risk_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get BERT embeddings
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = bert_output.pooler_output
        
        # Classify behaviors
        behavior_probs = self.behavior_classifier(pooled_output)
        
        # Estimate risk
        risk_score = self.risk_estimator(pooled_output)
        
        return {
            'behavior_probabilities': behavior_probs,
            'risk_score': risk_score,
            'embeddings': pooled_output
        }