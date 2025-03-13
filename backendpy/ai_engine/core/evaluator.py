from typing import Dict, Any, List
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

class ModelEvaluator:
    def __init__(self, model_manager, config: Dict[str, Any]):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(features)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    async def evaluate_predictions(
        self,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate prediction quality"""
        try:
            metrics = {}
            
            # Evaluate threat detection
            if 'is_threat' in predictions and 'is_threat' in ground_truth:
                metrics['threat_accuracy'] = int(
                    predictions['is_threat'] == ground_truth['is_threat']
                )
            
            # Evaluate anomaly detection
            if 'is_anomaly' in predictions and 'is_anomaly' in ground_truth:
                metrics['anomaly_accuracy'] = int(
                    predictions['is_anomaly'] == ground_truth['is_anomaly']
                )
            
            # Calculate confidence correlation
            if 'confidence' in predictions and 'confidence' in ground_truth:
                metrics['confidence_correlation'] = np.corrcoef(
                    [predictions['confidence']],
                    [ground_truth['confidence']]
                )[0, 1]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {e}")
            return {}