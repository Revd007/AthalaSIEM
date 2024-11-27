from typing import Dict, Any, Optional, List
import torch
import logging
from ..models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from ..models.threat_detector import ThreatDetector
from ..models.model_factory import AIModelFactory

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
            'threat_detector': ThreatDetector
        }
        
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models"""
        try:
            # Initialize anomaly detector
            if 'anomaly_detector' in self.config:
                self.models['anomaly_detector'] = AnomalyDetector(
                    **self.config['anomaly_detector']
                ).to(self.device)
            
            # Initialize VAE
            if 'vae' in self.config:
                self.models['vae'] = VariationalAutoencoder(
                    **self.config['vae']
                ).to(self.device)
            
            # Initialize threat detector
            if 'threat_detector' in self.config:
                self.models['threat_detector'] = ThreatDetector(
                    self.config['threat_detector']
                ).to(self.device)
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def get(self, model_type: str) -> Optional[torch.nn.Module]:
        if model_type not in self.models:
            self.models[model_type] = self.model_factory.create_model(model_type)
        return self.models.get(model_type)

    def get_model(self, model_type: str) -> Optional[torch.nn.Module]:
        """Get a specific model by type"""
        return self.models.get(model_type)

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