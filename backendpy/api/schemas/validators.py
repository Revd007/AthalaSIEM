from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID

class AlertCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    severity: str = Field(..., regex='^(low|medium|high|critical)$')
    source: str
    tags: Optional[List[str]] = []

    @validator('severity')
    def validate_severity(cls, v):
        allowed = ['low', 'medium', 'high', 'critical']
        if v.lower() not in allowed:
            raise ValueError(f'Severity must be one of {allowed}')
        return v.lower()

class EventCreate(BaseModel):
    event_type: str = Field(..., min_length=3)
    source: str
    description: str
    raw_data: dict
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

    @validator('event_type')
    def validate_event_type(cls, v):
        allowed_types = ['system', 'security', 'application', 'network']
        if v.lower() not in allowed_types:
            raise ValueError(f'Event type must be one of {allowed_types}')
        return v.lower()