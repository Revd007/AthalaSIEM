from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator
from datetime import datetime
import logging

class EventData(BaseModel):
    timestamp: datetime
    source: str
    event_type: str
    severity: int
    message: str
    raw_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    @validator('severity')
    def validate_severity(cls, v):
        if not 0 <= v <= 5:
            raise ValueError('Severity must be between 0 and 5')
        return v

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_stats = {
            'total_processed': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': {}
        }

    async def validate_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate single event data"""
        try:
            self.validation_stats['total_processed'] += 1
            validated_data = EventData(**event_data)
            self.validation_stats['valid_records'] += 1
            return validated_data.model_dump()
        except Exception as e:
            self._record_validation_error(str(e))
            self.validation_stats['invalid_records'] += 1
            return None

    def _record_validation_error(self, error: str):
        """Track validation errors for monitoring"""
        if error not in self.validation_stats['validation_errors']:
            self.validation_stats['validation_errors'][error] = 0
        self.validation_stats['validation_errors'][error] += 1