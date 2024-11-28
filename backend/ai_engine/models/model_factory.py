from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn
from pathlib import Path
import logging

from ai_engine.models.behavior_analyzer import BehaviorAnalyzer
from ai_engine.models.pattern_recognizer import PatternRecognizer
from ai_engine.models.risk_assessor import RiskAssessor
from .threat_detections import ThreatDetector
from .anomaly_detector import AnomalyDetector, VariationalAutoencoder
from .base_model import BaseModel

class AIModelFactory:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
    def _initialize_base_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize base configuration for all models"""
        base_config = {
            'version': '1.0',
            'anomaly_threshold': 0.8,
            'threat_threshold': 0.7
        }
        return {**base_config, **model_config}
        
    def create_model(self, model_type: str) -> Optional[BaseModel]:
        """Create a new model instance"""
        try:
            model_config = self.model_manager.config.get(model_type, {})
            base_config = self._initialize_base_config(model_config)
            
            if model_type == 'threat_detector':
                if not model_config:
                    model_config = {
                        'input_dim': 512,
                        'hidden_dim': 256,
                        'num_classes': 2
                    }
                return ThreatDetector(model_config)
                
            elif model_type == 'anomaly_detector':
                if not model_config:
                    model_config = {
                        'input_dim': 512,
                        'hidden_dim': 256
                    }
                return AnomalyDetector({**base_config, **model_config})
                
            elif model_type == 'vae':
                if not model_config:
                    model_config = {
                        'input_dim': 512,
                        'hidden_dims': [256, 128],
                        'latent_dim': 64
                    }
                return VariationalAutoencoder(**model_config)
                
            elif model_type == 'behavior_analyzer':
                if not model_config:
                    model_config = {
                        'bert_model': 'bert-base-uncased',
                        'num_behaviors': 10,
                        'hidden_dim': 768
                    }
                return BehaviorAnalyzer(**model_config)
                
            elif model_type == 'pattern_recognizer':
                if not model_config:
                    model_config = {
                        'input_dim': 512,
                        'hidden_dim': 256,
                        'num_layers': 3,
                        'dropout': 0.2
                    }
                return PatternRecognizer(**model_config)
                
            elif model_type == 'risk_assessor':
                if not model_config:
                    model_config = {
                        'input_dim': 256,
                        'hidden_dims': [128, 64, 32],
                        'num_risk_factors': 5
                    }
                return RiskAssessor(**model_config)
                
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating model {model_type}: {e}")
            return None