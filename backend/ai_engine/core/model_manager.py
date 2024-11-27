import torch
import logging
from typing import *
from ..models.anomaly_detector import VariationalAutoencoder
from ..models.threat_detector import ThreatDetector

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize models
        self.anomaly_detector = None
        self.threat_detector = None
        self._initialize_models()
        
    def _initialize_models(self):
        try:
            # Initialize anomaly detector
            anomaly_config = self.config.get('anomaly_detector', {})
            self.anomaly_detector = VariationalAutoencoder(
                input_dim=anomaly_config.get('input_dim', 256),
                hidden_dims=anomaly_config.get('hidden_dims', [128, 64]),
                latent_dim=anomaly_config.get('latent_dim', 32)
            ).to(self.device)
            
            # Initialize threat detector
            threat_config = self.config.get('threat_detector', {})
            self.threat_detector = ThreatDetector(
                config=threat_config
            ).to(self.device)
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
            
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert inputs to tensors
            inputs = torch.tensor(features['data'], dtype=torch.float32).to(self.device)
            
            # Get predictions from both models
            with torch.no_grad():
                anomaly_output = self.anomaly_detector(inputs)
                threat_output = self.threat_detector(inputs)
                
            return {
                'anomaly_score': anomaly_output[0].cpu().numpy(),
                'threat_score': threat_output.cpu().numpy()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def get_model(self, model_name=None):
        """Get a specific model or all models if no name is provided"""
        if model_name:
            return self.models.get(model_name)
        return self.models
    
    def register_model(self, name, model):
        """Register a model with the manager"""
        self.models[name] = model
    
    def get_enabled_models(self) -> List[str]:
        """Return a list of enabled model types"""
        enabled_models = []
        
        # Add models that are initialized and enabled
        if self.anomaly_detector is not None:
            enabled_models.append('anomaly_detector')
        if self.threat_detector is not None:
            enabled_models.append('threat_detector')
            
        # Add any additional models from the models dictionary
        enabled_models.extend([
            model_name for model_name, model in self.models.items()
            if model is not None
        ])
        
        return list(set(enabled_models))  # Remove any duplicates