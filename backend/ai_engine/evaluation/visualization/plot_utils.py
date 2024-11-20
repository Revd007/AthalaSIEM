import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import numpy as np
import pandas as pd

class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            labels: List[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels)
            plt.yticks(np.arange(len(labels)) + 0.5, labels)
        return plt.gcf()
    
    @staticmethod
    def plot_metrics_history(metrics_history: Dict[str, List[float]]) -> plt.Figure:
        """Plot training metrics history"""
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        return plt.gcf()
    
    @staticmethod
    def plot_detection_timeline(timestamps: List[float],
                              predictions: List[int],
                              actual: List[int] = None) -> plt.Figure:
        """Plot detection timeline"""
        plt.figure(figsize=(15, 5))
        plt.plot(timestamps, predictions, label='Predictions', marker='o')
        if actual:
            plt.plot(timestamps, actual, label='Actual', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Detection')
        plt.legend()
        plt.grid(True)
        return plt.gcf()