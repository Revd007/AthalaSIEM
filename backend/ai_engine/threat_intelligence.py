from typing import Dict, Any, Optional
import logging
from .models.threat_detections import ThreatDetector
from .core.model_manager import ModelManager
from .processors.feature_engineering import FeatureEngineer

class ThreatIntelligence:
    def __init__(self, model_manager: ModelManager):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.feature_engineer = FeatureEngineer()

    async def analyze_threats(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threats in Windows events"""
        try:
            # Extract features
            features = self.feature_engineer.extract_features(event)
            
            # Get threat analysis results
            results = await self.model_manager.analyze_threats(features)
            
            return {
                'is_threat': results['is_threat'],
                'threat_score': results['threat_score'],
                'indicators': results['indicators'],
                'confidence': results['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Threat analysis error: {e}")
            return {
                'is_threat': False,
                'threat_score': 0.0,
                'indicators': [],
                'confidence': 0.0,
                'error': str(e)
            }