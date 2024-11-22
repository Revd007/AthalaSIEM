from sqlalchemy import Boolean, Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.sql import func
import uuid
from .base import Base

class PlaybookRun(Base):
    __tablename__ = "playbook_runs"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    alert_id = Column(UNIQUEIDENTIFIER, ForeignKey('alerts.id'))
    playbook_id = Column(String)
    status = Column(String(50))  # running, completed, failed, cancelled
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<PlaybookRun {self.playbook_id} ({self.status})>"

class PlaybookTemplate(Base):
    __tablename__ = "playbook_templates"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    content = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<PlaybookTemplate {self.name}>"