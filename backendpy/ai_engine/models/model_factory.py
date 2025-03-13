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
from .unified_threat_detector import UnifiedThreatDetector
from .unified_anomaly_detector import UnifiedAnomalyDetector

class AIModelFactory:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Tambahkan definisi model_classes
        self.model_classes = {
            'anomaly_detector': AnomalyDetector,
            'vae': VariationalAutoencoder,
            'threat_detector': ThreatDetector,
            'behavior_analyzer': BehaviorAnalyzer,
            'pattern_recognizer': PatternRecognizer,
            'risk_assessor': RiskAssessor,
            'unified_threat_detector': UnifiedThreatDetector,
            'unified_anomaly_detector': UnifiedAnomalyDetector
        }
        
    def _initialize_base_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize base configuration for all models"""
        base_config = {
            'version': '1.0',
            'anomaly_threshold': 0.8,
            'threat_threshold': 0.7
        }
        return {**base_config, **model_config}
        
    def create_model(self, model_type: str, **kwargs) -> Optional[torch.nn.Module]:
        """Create a model instance"""
        try:
            if model_type not in self.model_classes:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
            
            # Get config from model settings
            model_config = self.model_manager.config.get('model_settings', {}).get(model_type, {}).copy()
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_config['device'] = device
            
            # Merge with kwargs
            model_config.update(kwargs)
            
            # Validate required fields
            if model_type == 'unified_threat_detector':
                required_fields = ['input_dim', 'hidden_dim']
                for field in required_fields:
                    if field not in model_config:
                        raise ValueError(f"Missing required config field for {model_type}: {field}")
            
            # Create model
            model = self.model_classes[model_type](model_config)
            model.to(device)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model {model_type}: {e}")
            return None