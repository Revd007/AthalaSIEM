from typing import Dict, Any, Optional
import torch
import logging
from ..models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from ..models.threat_detector import ThreatDetector

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self._initialize_models()
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def _initialize_models(self):
        """Initialize all required models"""
        try:
            # Initialize Anomaly Detector
            self.models['anomaly_detector'] = AnomalyDetector(
                input_dim=512,
                hidden_dim=256
            ).to(self.device)

            # Initialize VAE
            self.models['vae'] = VariationalAutoencoder(
                input_dim=512,
                hidden_dims=[256, 128],
                latent_dim=64
            ).to(self.device)

            # Initialize Threat Detector
            self.models['threat_detector'] = ThreatDetector({
                'input_dim': 512,
                'hidden_dim': 256,
                'num_classes': 2
            }).to(self.device)

        except Exception as e:
            self.logger.error(f"Error in model initialization: {e}")
            raise

    def get(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get a model by name"""
        try:
            return self.models.get(model_name)
        except Exception as e:
            self.logger.error(f"Error getting model {model_name}: {e}")
            return None

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