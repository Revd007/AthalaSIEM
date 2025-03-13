from typing import Dict, List, Any
import numpy as np
import torch
from scipy.stats import entropy

class CustomMetrics:
    @staticmethod
    def calculate_prediction_confidence(y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate prediction confidence metrics"""
        confidence_mean = np.mean(np.max(y_pred_proba, axis=1))
        confidence_std = np.std(np.max(y_pred_proba, axis=1))
        
        return {
            'mean_confidence': confidence_mean,
            'std_confidence': confidence_std
        }
    
    @staticmethod
    def calculate_prediction_entropy(y_pred_proba: np.ndarray) -> float:
        """Calculate prediction entropy as uncertainty measure"""
        return np.mean([entropy(pred) for pred in y_pred_proba])
    
    @staticmethod
    def calculate_detection_latency(timestamps: List[float], 
                                  predictions: List[int]) -> Dict[str, float]:
        """Calculate detection latency metrics"""
        detection_times = []
        current_sequence = []
        
        for t, pred in zip(timestamps, predictions):
            if pred == 1:  # Anomaly detected
                if not current_sequence:
                    current_sequence.append(t)
                else:
                    detection_times.append(t - current_sequence[0])
                    current_sequence = [t]
                    
        return {
            'mean_detection_time': np.mean(detection_times),
            'max_detection_time': np.max(detection_times),
            'min_detection_time': np.min(detection_times)
        }