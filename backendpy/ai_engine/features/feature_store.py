from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.connection import Base
import logging
from contextlib import asynccontextmanager

class FeatureSet(Base):
    __tablename__ = "feature_sets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    schema = Column(JSON)
    meta_info = Column(JSON)  # Changed from 'metadata' to 'meta_info'

class Feature(Base):
    __tablename__ = "features"
    
    id = Column(Integer, primary_key=True)
    feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))
    name = Column(String)
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_info = Column(JSON, nullable=True)  # Changed from 'metadata' to 'meta_info'

class FeatureStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._feature_sets: Dict[str, FeatureSet] = {}

    @asynccontextmanager
    async def _db_transaction(self, db: AsyncSession):
        """Context manager for database transactions"""
        try:
            yield
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise e

    async def create_feature_set(
        self,
        db: AsyncSession,
        name: str,
        schema: Dict[str, Any],
        meta_info: Optional[Dict[str, Any]] = None
    ) -> FeatureSet:
        """Create new feature set"""
        try:
            # Check if feature set already exists
            result = await db.execute(
                select(FeatureSet).where(FeatureSet.name == name)
            )
            existing_set = result.scalar_one_or_none()
            if existing_set:
                return existing_set

            feature_set = FeatureSet(
                name=name,
                version="1.0",
                schema=schema,
                meta_info=meta_info or {}
            )
            
            async with self._db_transaction(db):
                db.add(feature_set)
                await db.flush()
                await db.refresh(feature_set)
                
                self._feature_sets[name] = feature_set
                return feature_set

        except Exception as e:
            self.logger.error(f"Error creating feature set: {str(e)}")
            raise

    async def add_features(
        self,
        db: AsyncSession,
        feature_set_name: str,
        features: Dict[str, Union[float, List[float]]],
        meta_info: Optional[Dict[str, Any]] = None
    ) -> List[Feature]:
        """Add features to feature set"""
        try:
            feature_set = self._feature_sets.get(feature_set_name)
            if not feature_set:
                result = await db.execute(
                    select(FeatureSet).where(FeatureSet.name == feature_set_name)
                )
                feature_set = result.scalar_one_or_none()
                if not feature_set:
                    raise ValueError(f"Feature set {feature_set_name} not found")
                self._feature_sets[feature_set_name] = feature_set

            feature_records = []
            current_time = datetime.utcnow()

            for name, value in features.items():
                if isinstance(value, (list, np.ndarray)):
                    for v in value:
                        feature = Feature(
                            feature_set_id=feature_set.id,
                            name=name,
                            value=float(v),
                            timestamp=current_time,
                            meta_info=meta_info
                        )
                        feature_records.append(feature)
                else:
                    feature = Feature(
                        feature_set_id=feature_set.id,
                        name=name,
                        value=float(value),
                        timestamp=current_time,
                        meta_info=meta_info
                    )
                    feature_records.append(feature)
            
            async with self._db_transaction(db):
                db.add_all(feature_records)
                await db.flush()
                
                await self._update_cache(feature_set_name, features, current_time)
                
                return feature_records

        except Exception as e:
            self.logger.error(f"Error adding features: {str(e)}")
            raise

    async def get_features(
        self,
        db: AsyncSession,
        feature_set_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Feature]:
        """Get features from feature set with optional time range"""
        try:
            query = select(Feature).join(FeatureSet).where(FeatureSet.name == feature_set_name)
            
            if start_time:
                query = query.where(Feature.timestamp >= start_time)
            if end_time:
                query = query.where(Feature.timestamp <= end_time)
                
            result = await db.execute(query)
            return result.scalars().all()

        except Exception as e:
            self.logger.error(f"Error retrieving features: {str(e)}")
            raise

    async def _update_cache(
        self,
        feature_set_name: str,
        features: Dict[str, Any],
        timestamp: datetime
    ):
        """Update feature cache with timestamp"""
        if feature_set_name not in self._cache:
            self._cache[feature_set_name] = {}
            
        for name, value in features.items():
            self._cache[feature_set_name][name] = {
                'value': value,
                'timestamp': timestamp
            }