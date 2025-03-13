import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel

class ThreatDetector(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        # Pastikan config memiliki semua field yang diperlukan
        default_config = {
            'input_dim': 512,
            'hidden_dim': 256,
            'num_classes': 2,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'version': '1.0',
            'anomaly_threshold': 0.8,
            'threat_threshold': 0.7
        }
        
        # Merge config dengan default
        config = {**default_config, **config}
        
        # Panggil parent constructor
        super().__init__(config)
        
        # Simpan parameter sebagai instance variables
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        
        # Definisikan layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        # Pindahkan ke device yang sesuai
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Konversi input ke tensor jika bukan
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
            # Pindahkan ke device yang sama
            x = x.to(self.device)
            
            # Handle dimensi input
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            # Validasi ukuran input
            if x.size(-1) != self.input_dim:
                return torch.zeros(
                    1,  # batch size
                    self.num_classes,  # output classes
                    dtype=torch.float32,
                    device=self.device
                )
            
            return self.layers(x)
            
        except Exception as e:
            self.logger.error(f"Error in threat detector forward pass: {e}")
            return torch.zeros(
                1,  # batch size
                self.num_classes,  # output classes
                dtype=torch.float32,
                device=self.device
            )
