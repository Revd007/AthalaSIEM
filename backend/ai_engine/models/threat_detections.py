import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np

class ThreatDetector(nn.Module):
    def __init__(self, num_patterns: int = 10, num_behaviors: int = 5):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.threat_head = nn.Linear(768, 1)
        self.pattern_head = nn.Linear(768, num_patterns)
        self.behavior_head = nn.Linear(768, num_behaviors)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        features = self.transformer(x)
        
        return {
            'threat_score': torch.sigmoid(self.threat_head(features.mean(1))),
            'patterns': torch.softmax(self.pattern_head(features.mean(1)), dim=-1),
            'behaviors': torch.softmax(self.behavior_head(features.mean(1)), dim=-1)
        }