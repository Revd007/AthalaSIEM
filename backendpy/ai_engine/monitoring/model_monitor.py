from typing import Dict, Any, List
import torch
import numpy as np
from datetime import datetime
import logging
from ..core.model_manager import ModelManager
from ..evaluation.metrics.accuracy_metrics import AccuracyMetrics

class ModelMonitor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_thresholds = {
            'accuracy': 0.85,
            'f1_score': 0.80,
            'false_positive_rate': 0.15
        }

    async def monitor_model_health(self, model_name: str) -> Dict[str, Any]:
        try:
            model = self.model_manager.get(model_name)
            if not model:
                return {"status": "error", "message": f"Model {model_name} not found"}

            metrics = await self._calculate_model_metrics(model)
            self._update_metrics_history(model_name, metrics)
            health_status = self._evaluate_model_health(metrics)

            return {
                "status": "success",
                "model_name": model_name,
                "health_status": health_status,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error monitoring model {model_name}: {e}")
            return {"status": "error", "message": str(e)}

    def _evaluate_model_health(self, metrics: Dict[str, float]) -> str:
        if (metrics['accuracy'] < self.performance_thresholds['accuracy'] or
            metrics['f1_score'] < self.performance_thresholds['f1_score'] or
            metrics['false_positive_rate'] > self.performance_thresholds['false_positive_rate']):
            return "degraded"
        return "healthy"

    async def get_monitoring_report(self) -> Dict[str, Any]:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": {}
        }
        
        for model_name in self.model_manager.get_enabled_models():
            health_status = await self.monitor_model_health(model_name)
            report["models"][model_name] = health_status

        return report