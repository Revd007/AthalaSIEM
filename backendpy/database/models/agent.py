from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from .base import Base
from ..enums import AgentType, CollectorType, AgentStatus

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    type = Column(Enum(AgentType, name='agent_type'), nullable=False)
    collector_type = Column(Enum(CollectorType, name='collector_type'), nullable=False)
    status = Column(Enum(AgentStatus, name='agent_status'), nullable=False, default=AgentStatus.INACTIVE)
    ip_address = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    use_ssl = Column(Boolean, default=True)
    api_key = Column(String, unique=True, nullable=False)
    collector_config = Column(JSON, default={})
    enabled_sources = Column(JSON, default=[])
    filters = Column(JSON, default={})
    last_heartbeat = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Agent {self.name} ({self.type})>"