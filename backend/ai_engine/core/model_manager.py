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
        self.model_name = "Donquixote Athala"
        self.models = {}
        self.adaptive_learner = AdaptiveLearner(config)
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Threat Detection Model
            self.models['threat'] = ThreatDetector(
                num_classes=self.config.get('num_threat_classes', 5),
                model_name=self.config.get('bert_model', 'bert-base-uncased')
            )
            
            # Anomaly Detection Model
            self.models['anomaly'] = VariationalAutoencoder(
                input_dim=self.config.get('input_dim', 128),
                latent_dim=self.config.get('latent_dim', 32),
                hidden_dims=self.config.get('hidden_dims', [64, 32])
            )
            
            # Load pre-trained weights if available
            self._load_pretrained_weights()
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

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