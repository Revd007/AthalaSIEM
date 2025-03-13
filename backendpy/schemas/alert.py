from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class AlertStatus(str, Enum):
    NEW = "new"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

class AlertBase(BaseModel):
    title: str
    description: str
    severity: int
    source_event_id: int
    ai_analysis: Optional[Dict[str, Any]] = None

class AlertCreate(AlertBase):
    status: AlertStatus = AlertStatus.NEW
    assigned_to_id: Optional[int] = None

class Alert(AlertBase):
    id: int
    timestamp: datetime
    status: AlertStatus
    assigned_to_id: Optional[int]
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]

    class Config:
        from_attributes = True