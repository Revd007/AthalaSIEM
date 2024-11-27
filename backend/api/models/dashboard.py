from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class SystemMetrics(BaseModel):
    cpu: float
    memory: float
    storage: float
    network: float
    networkUsage: float

class Event(BaseModel):
    id: str
    title: str
    severity: str
    source: str
    timestamp: datetime
    status: str
    ai_analysis: Optional[Dict] = None

class EventsBySeverity(BaseModel):
    critical: int
    warning: int
    normal: int
    high: int
    medium: int
    low: int
    info: int
    error: int

class ChartDataPoint(BaseModel):
    timestamp: datetime
    value: int
    category: str

class EventsOverview(BaseModel):
    total: int
    by_severity: EventsBySeverity
    chart_data: List[ChartDataPoint]
    recent_events: List[Event]