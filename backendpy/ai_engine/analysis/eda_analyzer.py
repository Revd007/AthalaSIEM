import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

class EDAAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_window = timedelta(hours=1)
        self.min_samples = 100

    async def analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform exploratory data analysis on events"""
        try:
            if not events:
                return {}

            df = pd.DataFrame(events)
            
            # Basic statistics
            stats = {
                'event_count': len(df),
                'unique_sources': df['source'].nunique(),
                'severity_distribution': df['severity'].value_counts().to_dict(),
                'temporal_patterns': self._analyze_temporal_patterns(df),
                'source_analysis': self._analyze_sources(df),
                'correlation_matrix': self._calculate_correlations(df),
                'anomaly_scores': self._calculate_anomaly_scores(df)
            }

            return stats
        except Exception as e:
            self.logger.error(f"Error in EDA analysis: {e}")
            return {}

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in events"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        return {
            'hourly_distribution': df['hour'].value_counts().to_dict(),
            'event_frequency': len(df) / self.analysis_window.total_seconds(),
            'peak_hours': df.groupby('hour').size().nlargest(3).index.tolist()
        }

    def _analyze_sources(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze event sources and their characteristics"""
        source_stats = df.groupby('source').agg({
            'severity': ['mean', 'max'],
            'event_type': 'nunique'
        }).to_dict()

        return {
            'source_stats': source_stats,
            'critical_sources': df[df['severity'] <= 1]['source'].unique().tolist()
        }