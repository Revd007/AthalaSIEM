from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List, Any
import uuid

class EventBase(BaseModel):
    source: str
    event_type: str
    severity: int
    message: str
    host: Optional[str] = None
    ip_address: Optional[str] = None

class EventCreate(EventBase):
    raw_data: Optional[Dict] = None

class EventResponse(EventBase):
    id: int
    timestamp: datetime
    status: Optional[str]
    
    class Config:
        orm_mode = True

class AlertBase(BaseModel):
    title: str
    description: str
    severity: int
    source: str

class AlertCreate(AlertBase):
    events: List[int] = []

class AlertResponse(AlertBase):
    id: int
    timestamp: datetime
    status: str
    events: List[EventResponse]

    class Config:
        orm_mode = True

class AlertUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[int] = None
    status: Optional[str] = None
    source: Optional[str] = None
    assigned_to_id: Optional[uuid.UUID] = None
    resolved_at: Optional[datetime] = None
    alert_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True