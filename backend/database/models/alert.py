from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
from .base import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    severity = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)
    source = Column(String(100))
    assigned_to_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime)
    alert_metadata = Column(String)  # JSON data

    # Relationships
    assigned_to = relationship("User", back_populates="alerts")
    events = relationship("Event", back_populates="alert")

    def __repr__(self):
        return f"<Alert {self.title}>"