import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from .base_model import BaseModel
import logging

class UnifiedAnomalyDetector(BaseModel):
    """
    Unified Anomaly Detector yang menggabungkan VAE dan traditional anomaly detection
    dengan multiple detection heads dan robust error handling
    """
    def __init__(self, config: Dict[str, Any]):
        self._validate_config(config)
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        
        try:
            # Core parameters
            self.input_dim = config['input_dim']
            self.hidden_dims = config.get('hidden_dims', [256, 128])
            self.latent_dim = config.get('latent_dim', 64)
            self.dropout_rate = config.get('dropout_rate', 0.2)
            
            # Build encoder
            self.encoder = self._build_encoder()
            
            # Build decoder
            self.decoder = self._build_decoder()
            
            # Multiple heads for different types of anomalies
            self.reconstruction_head = nn.Linear(self.input_dim, self.input_dim)
            self.anomaly_classifier = nn.Linear(self.latent_dim, 1)
            self.pattern_head = nn.Linear(self.latent_dim, config.get('num_patterns', 10))
            
            self.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing UnifiedAnomalyDetector: {e}")
            raise

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters"""
        required_fields = ['input_dim']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Mean and log variance for VAE
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network"""
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple outputs"""
        try:
            # Ensure input is on correct device and type
            x = x.to(self.device, dtype=torch.float32)
            
            # Encode
            encoded = self.encoder(x)
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
            
            # Reparameterize
            z = self.reparameterize(mu, log_var)
            
            # Decode
            decoded = self.decoder(z)
            reconstructed = self.reconstruction_head(decoded)
            
            # Get various outputs
            anomaly_score = torch.sigmoid(self.anomaly_classifier(z))
            patterns = torch.softmax(self.pattern_head(z), dim=-1)
            
            return {
                'reconstructed': reconstructed,
                'anomaly_score': anomaly_score,
                'patterns': patterns,
                'mu': mu,
                'log_var': log_var,
                'latent': z
            }
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            return self._get_zero_outputs()

    def _get_zero_outputs(self) -> Dict[str, torch.Tensor]:
        """Generate zero outputs for error cases"""
        return {
            'reconstructed': torch.zeros((1, self.input_dim), device=self.device),
            'anomaly_score': torch.zeros(1, device=self.device),
            'patterns': torch.zeros((1, self.pattern_head.out_features), device=self.device),
            'mu': torch.zeros((1, self.latent_dim), device=self.device),
            'log_var': torch.zeros((1, self.latent_dim), device=self.device),
            'latent': torch.zeros((1, self.latent_dim), device=self.device)
        }

    def loss_function(self, 
                     x: torch.Tensor,
                     outputs: Dict[str, torch.Tensor],
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """Calculate VAE loss with additional components"""
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstructed'], x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def predict(self, x: torch.Tensor, threshold: float = None) -> Dict[str, Any]:
        """High-level prediction method"""
        with torch.no_grad():
            outputs = self.forward(x)
            threshold = threshold or self.config.get('anomaly_threshold', 0.7)
            
            return {
                'is_anomaly': outputs['anomaly_score'].item() > threshold,
                'anomaly_score': outputs['anomaly_score'].item(),
                'detected_patterns': outputs['patterns'].tolist(),
                'reconstruction_error': F.mse_loss(outputs['reconstructed'], x).item()
            }