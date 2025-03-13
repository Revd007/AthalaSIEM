import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class AnomalyDetector(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        model_config = config.get('model_settings', {}).get('unified_anomaly_detector', {})
        
        # Get model parameters from config or use defaults
        self.input_dim = model_config.get('input_dim', 512)
        self.hidden_dims = model_config.get('hidden_dims', [256, 128])
        self.latent_dim = model_config.get('latent_dim', 64)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        
        # Build encoder
        encoder_layers = []
        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent = nn.Linear(current_dim, self.latent_dim)
        
        # Build decoder
        decoder_layers = []
        current_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        x = self.encoder(x)
        return self.latent(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode
        z = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        # Calculate reconstruction error
        recon_error = torch.mean((x - x_recon) ** 2, dim=1)
        
        # Calculate anomaly score (normalized reconstruction error)
        anomaly_score = torch.sigmoid(recon_error)
        
        return {
            'latent': z,
            'reconstruction': x_recon,
            'recon_error': recon_error,
            'anomaly_score': anomaly_score
        }

class VariationalAutoencoder(nn.Module):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        model_config = config.get('model_settings', {}).get('unified_anomaly_detector', {})
        
        # Get model parameters from config or use defaults
        self.input_dim = model_config.get('input_dim', 512)
        self.hidden_dims = model_config.get('hidden_dims', [256, 128])
        self.latent_dim = model_config.get('latent_dim', 64)
        
        # Build encoder
        modules = []
        in_features = self.input_dim
        
        # Build encoder
        for h_dim in self.hidden_dims:
            modules.append(nn.Linear(in_features, h_dim))
            modules.append(nn.ReLU())
            in_features = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        
        # Build decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1])
        
        hidden_dims_reversed = self.hidden_dims[::-1]
        
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i + 1]))
            modules.append(nn.ReLU())
            
        modules.append(nn.Linear(hidden_dims_reversed[-1], self.input_dim))
        
        self.decoder = nn.Sequential(*modules)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        
        # Calculate reconstruction error
        recon_error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        
        # Calculate KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return {
            'reconstruction': recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'recon_error': recon_error,
            'kl_divergence': kl_div
        }

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss