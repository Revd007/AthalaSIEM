from datetime import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model=None, models=None, config=None):
        self.models = models if models else {'default': model}
        self.config = config
        self.metrics = {}
        
    async def evaluate(self, test_loader):
        """Comprehensive model evaluation"""
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'confusion_matrix': None,
            'attention_analysis': [],
            'error_analysis': [],
            'performance_metrics': {}
        }
        
        # Evaluate predictions
        predictions, labels = await self._get_predictions(test_loader)
        
        # Calculate metrics
        results.update(self._calculate_metrics(predictions, labels))
        
        # Analyze attention patterns
        results['attention_analysis'] = await self._analyze_attention_patterns(test_loader)
        
        # Perform error analysis
        results['error_analysis'] = self._analyze_errors(predictions, labels, test_loader)
        
        # Measure performance
        results['performance_metrics'] = self._measure_performance(test_loader)
        
        # Generate visualizations
        self._generate_evaluation_plots(results)
        
        return results
    
    async def _get_predictions(self, data_loader):
        """Get model predictions"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(**batch)
                predictions = outputs.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _calculate_metrics(self, predictions, labels):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Additional metrics
        metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
        metrics['per_class_accuracy'] = self._calculate_per_class_accuracy(
            labels, predictions
        )
        
        return metrics
    
    async def _analyze_attention_patterns(self, data_loader):
        """Analyze model's attention patterns"""
        attention_patterns = []
        
        with torch.no_grad():
            for batch in data_loader:
                attention_weights = self.model.get_attention_weights(**batch)
                attention_patterns.append(self._analyze_attention(attention_weights))
        
        return attention_patterns
    
    def _analyze_errors(self, predictions, labels, data_loader):
        """Detailed error analysis"""
        error_indices = np.where(predictions != labels)[0]
        error_analysis = []
        
        for idx in error_indices:
            error_analysis.append({
                'index': idx,
                'true_label': labels[idx],
                'predicted_label': predictions[idx],
                'confidence': self._get_prediction_confidence(idx, data_loader),
                'feature_importance': self._analyze_feature_importance(idx, data_loader)
            })
        
        return error_analysis
    
    def _measure_performance(self, data_loader):
        """Measure model performance metrics"""
        start_time = time.time()
        memory_start = torch.cuda.memory_allocated()
        
        # Run inference
        with torch.no_grad():
            for batch in data_loader:
                _ = self.model(**batch)
        
        return {
            'inference_time': time.time() - start_time,
            'memory_usage': torch.cuda.memory_allocated() - memory_start,
            'throughput': len(data_loader.dataset) / (time.time() - start_time)
        }
    
    def _save_results(self, report: Dict[str, Any]):
        """Save evaluation results and visualizations"""
        results_path = Path(self.config['results_dir'])
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(results_path / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate and save visualizations
        self._generate_visualizations(results_path)
    
    def _generate_visualizations(self, save_path: Path):
        """Generate evaluation visualizations"""
        # Confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.savefig(save_path / 'confusion_matrix.png')
        plt.close()
        
        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(self.metrics['fpr'], self.metrics['tpr'])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curve')
        plt.savefig(save_path / 'roc_curve.png')
        plt.close()

        # Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(self.metrics['recall'], self.metrics['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(save_path / 'precision_recall.png')
        plt.close()

        # Feature importance plot
        if 'feature_importance' in self.metrics:
            plt.figure(figsize=(12, 6))
            features = list(self.metrics['feature_importance'].keys())
            importance = list(self.metrics['feature_importance'].values())
            plt.barh(features, importance)
            plt.xlabel('Importance Score')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(save_path / 'feature_importance.png')
            plt.close()