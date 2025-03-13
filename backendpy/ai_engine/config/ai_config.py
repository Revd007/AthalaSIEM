from typing import Dict, Any
import yaml
from pathlib import Path

class AIConfig:
    def __init__(self):
        self.config_path = Path("backend/ai_engine/config/ai_settings.yaml")
        self.current_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load AI configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
            return {
                'ai_enabled': True,
                'model_settings': {
                    'anomaly_detector': {
                        'input_dim': 128,
                        'hidden_dim': 64,
                        'latent_dim': 32
                    },
                    'threat_detector': {
                        'input_dim': 256,
                        'hidden_dim': 128,
                        'num_classes': 4
                    }
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            }
        except Exception as e:
            print(f"Error loading AI config: {e}")
            return {}

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        self.current_config.update(new_config)
        self._save_config()

    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.current_config, f)
        except Exception as e:
            print(f"Error saving AI config: {e}") 