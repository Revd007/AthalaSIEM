from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch

class AccuracyMetrics:
    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def calculate_auc_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate AUC-ROC score"""
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    @staticmethod
    def calculate_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate confusion matrix based metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_negative_rate': tn / (tn + fp),
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp),
            'true_positive_rate': tp / (tp + fn)
        }