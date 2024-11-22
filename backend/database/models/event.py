from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
from .base import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    alert_id = Column(UNIQUEIDENTIFIER, ForeignKey('alerts.id'))
    event_type = Column(String(50), nullable=False)
    source = Column(String(100))
    timestamp = Column(DateTime, nullable=False)
    severity = Column(Integer)
    message = Column(Text)
    raw_data = Column(String)  # JSON data
    processed_data = Column(String)  # JSON data
    created_at = Column(DateTime, default=func.now())

    # Relationships
    alert = relationship("Alert", back_populates="events")

    def __repr__(self):
        return f"<Event {self.event_type}>"