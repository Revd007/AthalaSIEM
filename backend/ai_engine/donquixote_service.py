from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import logging
from .models.threat_detections import ThreatDetector
from .models.anomaly_detector import AnomalyDetector
from .core.model_manager import ModelManager
import yaml
from pathlib import Path

class DonquixoteService:
    def __init__(self, config_path: Union[str, Path]):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            raise
            
        # Initialize model manager with loaded config
        try:
            self.model_manager = ModelManager(self.config)
        except Exception as e:
            self.logger.error(f"Error initializing model manager: {e}")
            raise