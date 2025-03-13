from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

class AccuracyMetrics:
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='weighted',
                zero_division=0
            )
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        except Exception as e:
            print(f"Error calculating basic metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

    @staticmethod
    def calculate_advanced_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calculate advanced classification metrics"""
        try:
            metrics = {}
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC if probabilities are provided
            if y_prob is not None:
                try:
                    if y_prob.shape[1] == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                    else:  # Multi-class
                        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['roc_auc'] = float(roc_auc)
                except Exception as e:
                    print(f"Error calculating ROC AUC: {e}")
                    metrics['roc_auc'] = 0.0
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
            metrics['per_class'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
            return {
                'confusion_matrix': [[0]],
                'roc_auc': 0.0,
                'per_class': {
                    'precision': [0.0],
                    'recall': [0.0],
                    'f1': [0.0],
                    'support': [0]
                }
            }

    @staticmethod
    def calculate_prediction_confidence(y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate prediction confidence metrics"""
        try:
            confidence_mean = np.mean(np.max(y_prob, axis=1))
            confidence_std = np.std(np.max(y_prob, axis=1))
            
            return {
                'mean_confidence': float(confidence_mean),
                'std_confidence': float(confidence_std)
            }
        except Exception as e:
            print(f"Error calculating confidence metrics: {e}")
            return {
                'mean_confidence': 0.0,
                'std_confidence': 0.0
            }