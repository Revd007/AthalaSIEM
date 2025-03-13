import torch
from typing import Dict, Any
import numpy as np

class DifficultyEstimator:
    def __init__(self):
        self.feature_weights = {
            'length': 0.3,
            'complexity': 0.4,
            'novelty': 0.3
        }
        
        self.complexity_patterns = []
        self.novelty_threshold = 0.7
    
    def estimate(self, input_data: Dict[str, torch.Tensor]) -> float:
        """Estimate task difficulty"""
        # Calculate different aspects of difficulty
        length_score = self._estimate_length_difficulty(input_data)
        complexity_score = self._estimate_complexity(input_data)
        novelty_score = self._estimate_novelty(input_data)
        
        # Combine scores
        total_difficulty = (
            self.feature_weights['length'] * length_score +
            self.feature_weights['complexity'] * complexity_score +
            self.feature_weights['novelty'] * novelty_score
        )
        
        return total_difficulty
    
    def _estimate_length_difficulty(self, input_data: Dict[str, torch.Tensor]) -> float:
        """Estimate difficulty based on input length"""
        if 'input_ids' in input_data:
            length = input_data['input_ids'].size(1)
            # Normalize length score
            return min(length / 512, 1.0)  # Assuming max length of 512
        return 0.5
    
    def _estimate_complexity(self, input_data: Dict[str, torch.Tensor]) -> float:
        """Estimate input complexity"""
        if 'input_ids' in input_data:
            # Calculate entropy of input
            probs = torch.softmax(input_data['input_ids'].float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            
            # Normalize entropy
            return min(entropy / 10, 1.0)
        return 0.5
    
    def _estimate_novelty(self, input_data: Dict[str, torch.Tensor]) -> float:
        """Estimate how novel/unique the input is"""
        if not self.complexity_patterns:
            return 1.0
            
        # Calculate similarity with known patterns
        similarities = []
        for pattern in self.complexity_patterns:
            sim = self._calculate_similarity(input_data, pattern)
            similarities.append(sim)
        
        # High novelty if input is different from known patterns
        max_similarity = max(similarities)
        return 1.0 - max_similarity