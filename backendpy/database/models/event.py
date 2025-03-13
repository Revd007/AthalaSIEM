from sqlalchemy import Column, String, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from .base import Base

class Event(Base):
    __tablename__ = 'events'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    source = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    severity = Column(Integer, nullable=False)
    message = Column(String, nullable=False)
    ai_analysis = Column(JSON)
    
    # Relationship
    alerts = relationship("Alert", back_populates="event")

    def __repr__(self):
        return f"<Event {self.event_type}>"