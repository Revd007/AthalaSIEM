from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from .base import Base

class PlaybookTemplate(Base):
    __tablename__ = 'playbook_templates'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    steps = Column(JSON, nullable=False)
    triggers = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PlaybookRun(Base):
    __tablename__ = 'playbook_runs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    playbook_id = Column(UUID(as_uuid=True), ForeignKey('playbook_templates.id'))
    alert_id = Column(UUID(as_uuid=True), ForeignKey('alerts.id'), nullable=True)
    status = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    result = Column(JSON)

    # Relationships
    playbook = relationship("PlaybookTemplate")
    alert = relationship("Alert")