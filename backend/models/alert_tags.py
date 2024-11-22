from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base

class AlertTags(Base):
    __tablename__ = "alert_tags"

    alert_id = Column(
        UNIQUEIDENTIFIER,  # Changed from INTEGER to UNIQUEIDENTIFIER
        ForeignKey('alerts.id', name='FK_alert_tags_alerts'),
        primary_key=True
    )
    tag_id = Column(
        UNIQUEIDENTIFIER,
        ForeignKey('tags.id', name='FK_alert_tags_tags'),
        primary_key=True
    )
    added_at = Column(DateTime, default=datetime.utcnow)
    added_by = Column(
        UNIQUEIDENTIFIER,
        ForeignKey('users.id', name='FK_alert_tags_users'),
        nullable=True
    )

    # Relationships
    alert = relationship("Alerts", back_populates="tags")
    tag = relationship("Tags", back_populates="alerts")
    user = relationship("Users")