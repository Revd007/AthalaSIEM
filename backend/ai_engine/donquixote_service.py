from typing import Dict, Any
import torch
import logging
from .models.threat_detections import ThreatDetector
from .models.anomaly_detector import AnomalyDetector
from .core.model_manager import ModelManager
import yaml
from pathlib import Path

class DonquixoteService:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Akses GPU setting dengan proper dictionary path
        use_gpu = (
            self.config.get('model_settings', {}).get('use_gpu', True) and 
            self.config.get('inference', {}).get('enable_gpu', True)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model_manager = ModelManager(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AI configuration from yaml file"""
        try:
            # Gunakan Path untuk handle path dengan benar
            base_path = Path(__file__).parent.parent  # ke root backend/
            config_file = base_path / config_path
            
            self.logger.info(f"Looking for config file at: {config_file}")
            
            if not config_file.exists():
                self.logger.warning(f"Config file not found at {config_file}, using default config")
                return self._get_default_config()
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.logger.info("AI configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'ai_enabled': True,
            'model': {
                'name': 'donquixote',
                'version': '1.0.0',
                'type': 'transformer',
                'embedding_dim': 768,
                'num_heads': 12,
                'num_layers': 6,
                'dropout': 0.1
            },
            'model_settings': {
                'use_lightweight_model': True,
                'enable_gpu': True,
                'model_precision': 'float32',
                'architecture': 'transformer',
                'embedding_dim': 768,
                'num_heads': 12,
                'num_layers': 6,
                'dropout': 0.1,
                'max_sequence_length': 512,
                'vocab_size': 30000,
                'use_gpu': True,
                'batch_size': 32
            },
            'inference': {
                'threshold_confidence': 0.85,
                'max_batch_size': 16,
                'timeout_seconds': 30,
                'enable_gpu': True
            },
            'features': {
                'threat_detection': True,
                'anomaly_detection': True,
                'behavior_analysis': True,
                'pattern_recognition': True
            }
        }

    async def analyze_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze event using AI model"""
        try:
            if not self.config.get('ai_enabled', True):
                return {
                    'status': 'skipped',
                    'analysis': None,
                    'message': 'AI analysis is disabled'
                }

            # Implement event analysis logic here
            analysis_result = await self.model_manager.analyze_event(event_data)
            
            return {
                'status': 'success',
                'analysis': analysis_result,
                'message': 'Analysis completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing event: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to analyze event'
            }