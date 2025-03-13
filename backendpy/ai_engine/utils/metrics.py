from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import json
from pathlib import Path

class MetricsTracker:
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_metrics = {}
        self.metrics_history = []
        
    def update_metrics(self, 
                      metrics: Dict[str, float],
                      step: int,
                      model_name: Optional[str] = None):
        """Update current metrics"""
        timestamp = datetime.now().isoformat()
        
        metrics_entry = {
            'timestamp': timestamp,
            'step': step,
            'model_name': model_name,
            'metrics': metrics
        }
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Add to history
        self.metrics_history.append(metrics_entry)
        
        # Save metrics
        self._save_metrics(metrics_entry)
    
    def get_metric_trend(self, 
                        metric_name: str,
                        window_size: int = 10) -> List[float]:
        """Get trend for specific metric"""
        values = [
            entry['metrics'].get(metric_name)
            for entry in self.metrics_history[-window_size:]
            if metric_name in entry['metrics']
        ]
        
        return values if values else []
    
    def calculate_improvement(self, 
                            metric_name: str,
                            baseline: Optional[float] = None) -> Dict[str, float]:
        """Calculate improvement in metric"""
        if not self.metrics_history:
            return {}
            
        current_value = self.current_metrics.get(metric_name)
        if current_value is None:
            return {}
            
        if baseline is None and len(self.metrics_history) > 1:
            baseline = self.metrics_history[0]['metrics'].get(metric_name)
            
        if baseline is None:
            return {}
            
        absolute_change = current_value - baseline
        relative_change = (absolute_change / baseline) * 100 if baseline != 0 else 0
        
        return {
            'baseline': baseline,
            'current': current_value,
            'absolute_change': absolute_change,
            'relative_change': relative_change
        }
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved"""
        if not self.metrics_history:
            return {}
            
        best_metrics = {}
        for entry in self.metrics_history:
            for metric_name, value in entry['metrics'].items():
                if metric_name not in best_metrics or value > best_metrics[metric_name]['value']:
                    best_metrics[metric_name] = {
                        'value': value,
                        'step': entry['step'],
                        'timestamp': entry['timestamp']
                    }
                    
        return best_metrics
    
    def _save_metrics(self, metrics_entry: Dict[str, Any]):
        """Save metrics to file"""
        # Daily metrics file
        date_str = datetime.now().strftime('%Y%m%d')
        metrics_file = self.save_dir / f'metrics_{date_str}.jsonl'
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
        
        # Save best metrics
        best_metrics = self.get_best_metrics()
        best_metrics_file = self.save_dir / 'best_metrics.json'
        
        with open(best_metrics_file, 'w') as f:
            json.dump(best_metrics, f, indent=2)