from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class EventBase(BaseModel):
    timestamp: datetime
    source: str
    event_type: str
    severity: int
    message: str
    ai_analysis: Optional[Dict[str, Any]] = None

class EventCreate(EventBase):
    pass

class Event(EventBase):
    id: int

    class Config:
        from_attributes = True  # This enables ORM mode