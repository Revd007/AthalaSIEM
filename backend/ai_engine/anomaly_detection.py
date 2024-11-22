from typing import Dict, Any, Optional
import logging
from .models.anomaly_detector import VariationalAutoencoder
from .core.model_manager import ModelManager
from .processors.feature_engineering import FeatureEngineer

class AnomalyDetector:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = FeatureEngineer()

    async def detect_anomalies(self, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert features to numpy array if needed
            processed_features = self.feature_engineer.process_features(features)
            
            # Get model prediction
            prediction = await self.model_manager.predict(processed_features)
            
            # Convert numpy values to Python primitives
            return {
                'is_anomaly': bool(prediction.get('is_anomaly', False)),
                'anomaly_score': float(prediction.get('anomaly_score', 0.0)),
                'confidence': float(prediction.get('confidence', 0.0))
            }
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    # Add compatibility method for older code
    async def analyze(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return await self.detect_anomalies(features)