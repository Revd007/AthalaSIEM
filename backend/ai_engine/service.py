from datetime import datetime
from typing import Dict, Any, Optional
import logging
from .models.threat_detections import ThreatDetector
from .models.anomaly_detector import AnomalyDetector
from .core.model_manager import ModelManager

class AIService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.threat_detector = ThreatDetector()
        self.anomaly_detector = AnomalyDetector()
        self.logger = logging.getLogger(__name__)

    async def analyze_threat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            threat_result = await self.threat_detector.detect(data)
            return {
                "status": "success",
                "result": threat_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Threat analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomaly_result = await self.anomaly_detector.analyze(data)
            return {
                "status": "success",
                "result": anomaly_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }