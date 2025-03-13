from typing import Dict, Any, Optional, List
import torch
import logging
from ..models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from ..models.threat_detector import ThreatDetector
from ..models.model_factory import AIModelFactory
from ..models.behavior_analyzer import BehaviorAnalyzer
from ..models.pattern_recognizer import PatternRecognizer
from ..models.risk_assessor import RiskAssessor

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_factory = AIModelFactory(self)
        
        # Initialize model classes
        self.model_classes = {
            'anomaly_detector': AnomalyDetector,
            'vae': VariationalAutoencoder,
            'threat_detector': ThreatDetector,
            'behavior_analyzer': BehaviorAnalyzer,
            'pattern_recognizer': PatternRecognizer,
            'risk_assessor': RiskAssessor
        }
        
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models"""
        try:
            model_settings = self.config.get('model_settings', {})
            if not model_settings:
                self.logger.error("No model settings found in config")
                return
            
            for model_type, settings in model_settings.items():
                if model_type in self.model_classes:
                    # Tambahkan device ke settings
                    settings['device'] = self.device
                    
                    # Buat model dengan settings yang lengkap
                    self.models[model_type] = self.model_factory.create_model(
                        model_type,
                        **settings
                    )
                    
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")

    def get(self, model_type: str) -> Optional[torch.nn.Module]:
        if model_type not in self.models:
            self.models[model_type] = self.model_factory.create_model(model_type)
        return self.models.get(model_type)

    def get_model(self, model_type: str) -> Optional[torch.nn.Module]:
        """Mengambil model berdasarkan tipe"""
        try:
            if model_type not in self.models:
                model = self.model_factory.create_model(model_type)
                if model is not None:
                    model = model.to(self.device)  # Memindahkan model ke GPU
                    self.models[model_type] = model
            return self.models.get(model_type)
        except Exception as e:
            self.logger.error(f"Error getting model '{model_type}': {e}")
            return None

    def get_default_model(self) -> Optional[torch.nn.Module]:
        """Get the default model (first available)"""
        if not self.models:
            return None
        return next(iter(self.models.values()))

    def get_all_models(self) -> Dict[str, torch.nn.Module]:
        """Get all initialized models"""
        return self.models

    def save_model(self, model_name: str, path: str):
        """Save a model to disk"""
        try:
            model = self.models.get(model_name)
            if model is not None:
                torch.save(model.state_dict(), path)
                self.logger.info(f"Model {model_name} saved successfully")
            else:
                self.logger.error(f"Model {model_name} not found")
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")

    def load_model(self, model_name: str, path: str):
        """Load a model from disk"""
        try:
            model = self.models.get(model_name)
            if model is not None:
                model.load_state_dict(torch.load(path))
                model.eval()
                self.logger.info(f"Model {model_name} loaded successfully")
            else:
                self.logger.error(f"Model {model_name} not found")
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model"""
        model = self.models.get(model_name)
        if model is None:
            return {"status": "not_found"}
        
        return {
            "status": "loaded",
            "device": str(next(model.parameters()).device),
            "training": model.training
        }

    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        return {
            name: self.get_model_status(name)
            for name in self.models
        }

    def get_enabled_models(self) -> List[str]:
        """Get a list of enabled model types"""
        enabled_models = []
        for model_name, model in self.models.items():
            if model is not None:
                enabled_models.append(model_name)
        return enabled_models