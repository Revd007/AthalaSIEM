import datetime
import torch
import numpy as nn
from typing import Dict, List, Any, Optional
from sklearn.cluster import DBSCAN
from collections import defaultdict

from backend.ai_engine.core.knowledge_graph import KnowledgeGraph
from backend.ai_engine.training.difficulty_estimator import DifficultyEstimator

class AdaptiveLearner:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.learning_patterns = defaultdict(list)
        self.pattern_importance = defaultdict(float)
        self.difficulty_estimator = DifficultyEstimator()
        self.knowledge_graph = KnowledgeGraph()
        
        # Hyperparameters
        self.min_confidence = config.get('min_confidence', 0.8)
        self.max_patterns = config.get('max_patterns', 1000)
        self.learning_rate_adjust = config.get('learning_rate_adjust', 0.1)
    
    async def learn_from_experience(self, 
                                  input_data: Dict[str, torch.Tensor], 
                                  output_data: Dict[str, torch.Tensor],
                                  feedback: Optional[Dict[str, Any]] = None):
        """Learn from each interaction adaptively"""
        # Analyze input complexity
        difficulty = self.difficulty_estimator.estimate(input_data)
        
        # Extract patterns
        patterns = self._extract_patterns(input_data, output_data)
        
        # Update knowledge graph
        self.knowledge_graph.update(patterns, feedback)
        
        # Adjust learning strategy
        learning_rate = self._adjust_learning_rate(difficulty)
        
        # Update model with new knowledge
        await self._update_model(patterns, learning_rate)
        
        # Prune outdated patterns
        self._prune_patterns()
    
    def _extract_patterns(self, 
                         input_data: Dict[str, torch.Tensor], 
                         output_data: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Extract learning patterns from input-output pairs"""
        patterns = []
        
        # Extract input features
        features = self.model.encoder(input_data)
        
        # Cluster similar patterns
        clustering = DBSCAN(eps=0.3, min_samples=2)
        cluster_labels = clustering.fit_predict(features.detach().cpu().numpy())
        
        # Group by clusters
        for label in set(cluster_labels):
            if label == -1:  # Skip noise
                continue
                
            cluster_mask = cluster_labels == label
            cluster_features = features[cluster_mask]
            
            pattern = {
                'centroid': torch.mean(cluster_features, dim=0),
                'variance': torch.var(cluster_features, dim=0),
                'frequency': sum(cluster_mask),
                'timestamp': datetime.utcnow()
            }
            patterns.append(pattern)
        
        return patterns

    async def _update_model(self, patterns: List[Dict[str, Any]], learning_rate: float):
        """Update model based on new patterns"""
        for pattern in patterns:
            # Generate synthetic examples from pattern
            synthetic_data = self._generate_synthetic_data(pattern)
            
            # Update model weights
            self.model.optimizer.optimizer.param_groups[0]['lr'] = learning_rate
            await self.model.optimizer.train_step(synthetic_data)
            
            # Update pattern importance
            self.pattern_importance[hash(pattern['centroid'].tostring())] += 1
    
    def _adjust_learning_rate(self, difficulty: float) -> float:
        """Adjust learning rate based on task difficulty"""
        base_lr = self.model.optimizer.optimizer.param_groups[0]['lr']
        
        if difficulty > 0.8:  # Hard task
            return base_lr * (1 + self.learning_rate_adjust)
        elif difficulty < 0.2:  # Easy task
            return base_lr * (1 - self.learning_rate_adjust)
        
        return base_lr
    
    def _generate_synthetic_data(self, pattern: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate synthetic training examples from pattern"""
        num_samples = 10
        noise_scale = 0.1
        
        # Generate samples around pattern centroid
        synthetic_samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(pattern['centroid']) * pattern['variance'] * noise_scale
            sample = pattern['centroid'] + noise
            synthetic_samples.append(sample)
        
        return {
            'input_ids': torch.stack(synthetic_samples),
            'attention_mask': torch.ones(num_samples, synthetic_samples[0].size(0))
        }
    
    def _prune_patterns(self):
        """Remove outdated or less important patterns"""
        if len(self.pattern_importance) > self.max_patterns:
            # Sort patterns by importance
            sorted_patterns = sorted(
                self.pattern_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep only top patterns
            self.pattern_importance = defaultdict(float)
            for pattern_hash, importance in sorted_patterns[:self.max_patterns]:
                self.pattern_importance[pattern_hash] = importance