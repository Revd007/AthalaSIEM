from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn
from pathlib import Path
import logging

from ai_engine.models.behavior_analyzer import BehaviorAnalyzer
from ai_engine.models.pattern_recognizer import PatternRecognizer
from ai_engine.models.risk_assessor import RiskAssessor
from .threat_detections import ThreatDetector
from .anomaly_detector import VariationalAutoencoder
from .base_model import BaseModel

class AIModelFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_registry = {
            'threat_detector': ThreatDetector,
            'anomaly_detector': VariationalAutoencoder,
            'behavior_analyzer': BehaviorAnalyzer,
            'pattern_recognizer': PatternRecognizer,
            'risk_assessor': RiskAssessor
        }
        
    def create_model(self, model_type: str) -> Optional[BaseModel]:
        """Create an AI model instance"""
        try:
            if model_type not in self.model_registry:
                raise ValueError(f"Unknown model type: {model_type}")
                
            model_class = self.model_registry[model_type]
            model_config = self.config.get(model_type, {})
            
            return model_class(**model_config)
            
        except Exception as e:
            self.logger.error(f"Error creating model {model_type}: {e}")
            return None