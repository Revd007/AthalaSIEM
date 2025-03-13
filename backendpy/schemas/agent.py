from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from database.models.agent import AgentType, CollectorType, AgentStatus

class AgentBase(BaseModel):
    name: str
    type: str = Field(
        ..., 
        description="Agent type", 
        pattern=f"^({'|'.join([t.value.replace('_collector', '') for t in AgentType])})$"
    )
    collector_type: str = Field(
        ..., 
        description="Collector type", 
        pattern=f"^({'|'.join([t.value for t in CollectorType])})$"
    )
    ip_address: str
    port: str
    use_ssl: bool = True
    collector_config: Optional[Dict[str, Any]] = {}
    enabled_sources: List[str] = []

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = Field(
        None, 
        pattern=f"^({'|'.join([t.value.replace('_collector', '') for t in AgentType])})$"
    )
    collector_type: Optional[str] = Field(
        None, 
        pattern=f"^({'|'.join([t.value for t in CollectorType])})$"
    )
    ip_address: Optional[str] = None
    port: Optional[str] = None
    use_ssl: Optional[bool] = None
    status: Optional[str] = Field(
        None, 
        pattern=f"^({'|'.join([t.value for t in AgentStatus])})$"
    )
    collector_config: Optional[Dict[str, Any]] = None
    enabled_sources: Optional[List[str]] = None

class AgentResponse(AgentBase):
    id: str
    status: str
    api_key: str
    last_heartbeat: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 