from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
from ..core.model_manager import ModelManager
from ..processors.feature_engineering import FeatureEngineer
from ..processors.data_normalization import DataNormalizer
from ..evaluation.metrics import AccuracyMetrics

class AIPipelineOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager(config)
        self.feature_engineer = FeatureEngineer()
        self.data_normalizer = DataNormalizer()
        self.metrics = AccuracyMetrics()
        
    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event through the AI pipeline"""
        try:
            # Normalize data
            normalized_event = self.data_normalizer.normalize_event(event)
            
            # Extract features
            features = self.feature_engineer.extract_features(normalized_event)
            
            # Process through AI models
            ai_analysis = await self.model_manager.process_event(features)
            
            # Enrich results
            enriched_results = await self._enrich_analysis(
                event, 
                normalized_event, 
                ai_analysis
            )
            
            # Update metrics
            self._update_metrics(enriched_results)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Error in AI pipeline: {e}")
            return self._create_error_response(e)
            
    async def _enrich_analysis(
        self,
        original_event: Dict[str, Any],
        normalized_event: Dict[str, Any],
        ai_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich AI analysis with additional context"""
        return {
            'event_id': original_event.get('id'),
            'timestamp': datetime.now().isoformat(),
            'original_event': original_event,
            'ai_analysis': ai_analysis,
            'risk_score': self._calculate_risk_score(ai_analysis),
            'recommendations': await self._generate_recommendations(ai_analysis),
            'confidence_metrics': {
                'threat_confidence': ai_analysis.get('threat', {}).get('confidence'),
                'anomaly_confidence': ai_analysis.get('anomaly', {}).get('confidence')
            }
        }