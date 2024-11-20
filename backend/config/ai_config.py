from typing import Dict, Any
import yaml
import os

class AIConfig:
    def __init__(self):
        self.config_path = "config/ai_settings.yaml"
        self.default_config = {
            'ai_enabled': False,  # Default disabled
            'resource_settings': {
                'max_memory_usage': 2048,  # MB
                'max_cpu_usage': 50,       # Percentage
                'batch_size': 32,
                'num_workers': 2
            },
            'model_settings': {
                'use_lightweight_model': True,
                'enable_gpu': False,
                'model_precision': 'float32'
            },
            'feature_toggles': {
                'anomaly_detection': True,
                'threat_detection': True,
                'adaptive_learning': False
            }
        }
        self.current_config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self.default_config
    
    def save_config(self):
        """Save current configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.current_config, f)