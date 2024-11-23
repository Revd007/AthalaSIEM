import uuid
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import *
from datetime import datetime
from uuid import UUID

class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    role: str = "user"
    model_config = ConfigDict(from_attributes=True)

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

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
    
    model_config = ConfigDict(from_attributes=True)

class AlertBase(BaseModel):
    title: str
    description: str
    severity: int
    source: str

class AlertCreate(AlertBase):
    events: List[int] = []

class AlertResponse(AlertBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime]
    events: List[EventResponse]
    
    model_config = ConfigDict(from_attributes=True)

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

class PlaybookTemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    content: Dict[str, Any]
    is_active: bool = True

class PlaybookTemplateCreate(PlaybookTemplateBase):
    pass

class PlaybookTemplateResponse(PlaybookTemplateBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime]
    created_by: Optional[UUID]

    class Config:
        from_attributes = True

class PlaybookRunResponse(BaseModel):
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    result: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    system_info: Dict[str, str]

class CPUMetrics(BaseModel):
    usage_percent: float
    count: int

class MemoryMetrics(BaseModel):
    total: int
    available: int
    used: int
    usage_percent: float

class DiskMetrics(BaseModel):
    total: int
    used: int
    free: int
    usage_percent: float

class NetworkMetrics(BaseModel):
    bytes_sent: int
    bytes_recv: int

class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    network: NetworkMetrics