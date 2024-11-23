from typing import Dict, Any, Optional
import torch
from .models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from .models.threat_detector import ThreatDetector
import logging

class DonquixoteService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default config if none provided
        default_config = {
            'anomaly_detector': {
                'input_dim': 512,
                'hidden_dims': [256, 128],
                'latent_dim': 64
            },
            'threat_detector': {
                'input_dim': 512,
                'embedding_dim': 768,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1
            }
        }
        
        # Use provided config or default
        self.config = config if config is not None else default_config
        
        try:
            # Initialize models
            self.anomaly_detector = VariationalAutoencoder(
                input_dim=self.config['anomaly_detector']['input_dim'],
                hidden_dims=self.config['anomaly_detector']['hidden_dims'],
                latent_dim=self.config['anomaly_detector']['latent_dim']
            ).to(self.device)
            
            self.threat_detector = ThreatDetector(
                config=self.config['threat_detector']
            ).to(self.device)
            
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    async def analyze_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert input data to tensor
            input_tensor = self._preprocess_data(event_data)
            
            # Get predictions from both models
            with torch.no_grad():
                anomaly_score = self.anomaly_detector(input_tensor)[0]
                threat_score = self.threat_detector(input_tensor)
            
            return {
                'anomaly_score': float(anomaly_score.mean().item()),
                'threat_score': float(threat_score.mean().item()),
                'is_anomaly': float(anomaly_score.mean().item()) > 0.5,
                'is_threat': float(threat_score.mean().item()) > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing event: {e}")
            return {
                'error': str(e),
                'anomaly_score': 0.0,
                'threat_score': 0.0,
                'is_anomaly': False,
                'is_threat': False
            }
    
    def _preprocess_data(self, event_data: Dict[str, Any]) -> torch.Tensor:
        # Implement your preprocessing logic here
        # This is a placeholder implementation
        return torch.randn(1, 512).to(self.device)