from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from datetime import datetime
import logging
from pathlib import Path
import json
import numpy as np
from ..models.threat_detections import ThreatDetector
from ..models.anomaly_detector import VariationalAutoencoder
from ..training.adaptive_learner import AdaptiveLearner

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        
        # Initialize models based on config
        if config.get('feature_toggles', {}).get('anomaly_detection'):
            self.init_anomaly_detector()
            
        if config.get('feature_toggles', {}).get('threat_detection'):
            self.init_threat_detector()
    
    def init_anomaly_detector(self):
        """Initialize anomaly detection model"""
        try:
            model_config = {
                'input_dim': 256,
                'latent_dim': 32,
                'anomaly_threshold': 0.8
            }
            self.models['anomaly_detector'] = VariationalAutoencoder(model_config)
        except Exception as e:
            self.logger.error(f"Error initializing anomaly detector: {e}")
    
    def init_threat_detector(self):
        """Initialize threat detection model"""
        try:
            model_config = {
                'num_classes': 5,
                'threat_threshold': 0.7
            }
            self.models['threat_detector'] = ThreatDetector(model_config)
        except Exception as e:
            self.logger.error(f"Error initializing threat detector: {e}")

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process security event through AI models"""
        try:
            # Preprocess event data
            features = self._extract_features(event)
            
            # Run threat detection
            threat_result = await self._detect_threats(features)
            
            # Run anomaly detection
            anomaly_result = await self._detect_anomalies(features)
            
            # Combine and analyze results
            analysis = self._combine_analysis(threat_result, anomaly_result)
            
            # Update adaptive learning
            await self.adaptive_learner.learn_from_event(event, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _detect_threats(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect threats using threat detection model"""
        with torch.no_grad():
            threat_model = self.models['threat']
            outputs = threat_model(features)
            
            return {
                'threat_level': outputs['threat_level'],
                'confidence': outputs['confidence'],
                'indicators': outputs['indicators'],
                'explanation': threat_model.explain_prediction(outputs)
            }

    async def _detect_anomalies(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies using VAE model"""
        with torch.no_grad():
            anomaly_model = self.models['anomaly']
            reconstruction, mu, logvar = anomaly_model(features)
            
            anomaly_score = anomaly_model.compute_anomaly_score(
                features, reconstruction, mu, logvar
            )
            
            return {
                'is_anomaly': anomaly_score > anomaly_model.threshold,
                'anomaly_score': float(anomaly_score),
                'reconstruction_error': float(torch.mean((features - reconstruction)**2))
            }

    async def detect_anomalies(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies using anomaly detection model"""
        try:
            if 'anomaly_detector' not in self.models:
                return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}
                
            with torch.no_grad():
                model = self.models['anomaly_detector']
                outputs = model(features)
                reconstruction_error = torch.mean((features - outputs['reconstruction'])**2)
                
                return {
                    'is_anomaly': reconstruction_error > model.threshold,
                    'anomaly_score': float(reconstruction_error),
                    'confidence': float(outputs.get('confidence', 0.8))
                }
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}

    async def analyze_threats(self, features: torch.Tensor) -> Dict[str, Any]:
        """Analyze threats using threat detection model"""
        try:
            if 'threat_detector' not in self.models:
                return {'is_threat': False, 'threat_score': 0.0, 'confidence': 0.0}
                
            with torch.no_grad():
                model = self.models['threat_detector']
                outputs = model(features)
                threat_score = outputs.get('threat_score', 0.0)
                
                return {
                    'is_threat': threat_score > model.threshold,
                    'threat_score': float(threat_score),
                    'indicators': outputs.get('indicators', []),
                    'confidence': float(outputs.get('confidence', 0.8))
                }
        except Exception as e:
            self.logger.error(f"Error in threat analysis: {e}")
            return {'is_threat': False, 'threat_score': 0.0, 'confidence': 0.0}