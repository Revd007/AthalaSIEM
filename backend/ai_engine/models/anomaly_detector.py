import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Encoder
        modules = []
        in_features = input_dim
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2)
                )
            )
            in_features = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder
        modules = []
        hidden_dims.reverse()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2)
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        z = self.decoder(z)
        return self.final_layer(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon_x, mu, log_var = self.forward(x)
        
        # Reconstruction probability
        recon_error = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Combined anomaly score
        anomaly_score = recon_error + kl_div
        return anomaly_score

class AnomalyDetector:
    def __init__(self, model: VariationalAutoencoder, threshold: float = 0.95):
        self.model = model
        self.threshold = threshold
        self.normal_scores = []
    
    def fit_threshold(self, normal_data: torch.Tensor):
        """Fit threshold based on normal data distribution"""
        with torch.no_grad():
            scores = self.model.get_anomaly_score(normal_data)
            self.threshold = np.percentile(scores.numpy(), 95)  # 95th percentile
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict if input is anomalous"""
        with torch.no_grad():
            anomaly_score = self.model.get_anomaly_score(x)
            is_anomaly = anomaly_score > self.threshold
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'threshold': self.threshold
            }