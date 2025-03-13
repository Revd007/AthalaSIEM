from typing import Dict, Any
import torch
import asyncio
import logging
from datetime import datetime

from ..evaluation.metrics.accuracy_metrics import AccuracyMetrics
from ..evaluation.metrics.custom_metrics import CustomMetrics

class ModelOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.accuracy_metrics = AccuracyMetrics()
        self.custom_metrics = CustomMetrics()

    async def process_event(
        self,
        features: Dict[str, Any],
        models: Dict[str, torch.nn.Module]
    ) -> Dict[str, Any]:
        """Process event through multiple models"""
        try:
            results = {}
            
            # Process through each model
            for model_name, model in models.items():
                with torch.no_grad():
                    output = await self._run_model(model, features)
                    results[model_name] = self._format_output(output, model_name)
            
            # Add metadata
            results['timestamp'] = datetime.utcnow().isoformat()
            results['processing_info'] = {
                'models_used': list(models.keys()),
                'feature_count': len(features)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model orchestration: {e}")
            raise

    async def _run_model(
        self,
        model: torch.nn.Module,
        features: Dict[str, Any]
    ) -> torch.Tensor:
        """Run a single model"""
        try:
            # Convert features to tensor if needed
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(
                    [list(features.values())],
                    dtype=torch.float32
                )
            
            return model(features)
            
        except Exception as e:
            self.logger.error(f"Error running model: {e}")
            raise

    def _format_output(
        self,
        output: torch.Tensor,
        model_name: str
    ) -> Dict[str, Any]:
        """Format model output"""
        try:
            # Convert to numpy for easier handling
            output_np = output.cpu().numpy()
            
            # Different formatting based on model type
            if model_name == 'threat':
                return {
                    'score': float(output_np[0][1]),  # Probability of threat
                    'confidence': float(output_np.max())
                }
            elif model_name == 'anomaly':
                return {
                    'score': float(output_np[0][0]),  # Anomaly score
                    'confidence': float(output_np.max())
                }
            else:
                return {
                    'raw_output': output_np.tolist(),
                    'confidence': float(output_np.max())
                }
                
        except Exception as e:
            self.logger.error(f"Error formatting output: {e}")
            return {
                'error': str(e),
                'score': 0.0,
                'confidence': 0.0
            }