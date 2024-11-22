from typing import Dict, Any, List
import torch
import numpy as np
import logging
from ..models.model_factory import AIModelFactory

class AIEnsembleManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_factory = AIModelFactory(config)
        self.models = self._initialize_models()
        self.weights = self._load_ensemble_weights()
        
    def _initialize_models(self) -> Dict[str, torch.nn.Module]:
        """Initialize all models in ensemble"""
        return {
            model_type: self.model_factory.create_model(model_type)
            for model_type in self.config['enabled_models']
        }
        
    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process event through ensemble"""
        try:
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                predictions[model_name] = await self._get_model_prediction(
                    model, event
                )
                
            # Combine predictions
            ensemble_result = self._combine_predictions(predictions)
            
            # Add confidence metrics
            ensemble_result['confidence'] = self._calculate_confidence(predictions)
            
            # Add explanation
            ensemble_result['explanation'] = self._generate_explanation(
                predictions, ensemble_result
            )
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble processing error: {e}")
            return self._create_error_response(e)