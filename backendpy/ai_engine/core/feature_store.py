from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from database.connection import Base

class FeatureStore:
    def __init__(self):
        self.features = {}
        self.metadata = {}

    async def add_features(
        self,
        session: AsyncSession,
        feature_set: str,
        features: Dict[str, Any],
        meta_info: Optional[Dict[str, Any]] = None
    ):
        """Store features with metadata"""
        timestamp = datetime.utcnow()
        
        if feature_set not in self.features:
            self.features[feature_set] = []
            self.metadata[feature_set] = []
            
        self.features[feature_set].append(features)
        self.metadata[feature_set].append({
            'timestamp': timestamp,
            'meta_info': meta_info or {}
        })

    async def get_features(
        self,
        feature_set: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get features with optional time range"""
        if feature_set not in self.features:
            return []
            
        features = []
        for feat, meta in zip(self.features[feature_set], self.metadata[feature_set]):
            timestamp = meta['timestamp']
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            features.append(feat)
            
        return features 