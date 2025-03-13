from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import hashlib
from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from database.connection import Base

class DataVersion(Base):
    __tablename__ = "data_versions"
    
    id = Column(Integer, primary_key=True)
    version_hash = Column(String, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_info = Column(JSON)  # Changed from 'metadata' to 'meta_info'
    feature_schema = Column(JSON)
    data_stats = Column(JSON)

class DataVersionManager:
    def __init__(self):
        self.current_version: Optional[str] = None
        self.version_cache: Dict[str, Dict] = {}
        
    async def create_version(
        self, 
        data: Dict[str, Any],
        meta_info: Dict[str, Any],  # Changed from 'metadata' to 'meta_info'
        db: AsyncSession
    ) -> str:
        """Create new data version"""
        try:
            version_hash = self._generate_hash(data)
            data_stats = self._calculate_stats(data)
            feature_schema = self._extract_schema(data)
            
            version = DataVersion(
                version_hash=version_hash,
                meta_info=meta_info,  # Changed parameter name
                feature_schema=feature_schema,
                data_stats=data_stats
            )
            
            db.add(version)
            await db.commit()
            await db.refresh(version)
            
            self.version_cache[version_hash] = {
                'meta_info': meta_info,  # Changed key name
                'feature_schema': feature_schema,
                'data_stats': data_stats,
                'timestamp': version.timestamp
            }
            
            self.current_version = version_hash
            return version_hash
            
        except Exception as e:
            await db.rollback()
            raise Exception(f"Error creating version: {str(e)}")

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate deterministic hash for data version"""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _calculate_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic statistics for data version"""
        stats = {
            'feature_count': len(data.keys()),
            'null_counts': {},
            'unique_counts': {},
            'data_types': {}
        }
        
        for key, value in data.items():
            if isinstance(value, list):
                stats['null_counts'][key] = sum(1 for v in value if v is None)
                stats['unique_counts'][key] = len(set(value))
                stats['data_types'][key] = str(type(value[0])) if value else 'unknown'
                
        return stats

    def _extract_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature schema from data"""
        schema = {}
        for key, value in data.items():
            if isinstance(value, list):
                schema[key] = {
                    'type': str(type(value[0])) if value else 'unknown',
                    'nullable': any(v is None for v in value),
                    'sample': value[0] if value else None
                }
        return schema