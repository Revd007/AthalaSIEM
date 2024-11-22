from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4

from ..database import Base

class Comments(Base):
    __tablename__ = "comments"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid4)
    alert_id = Column(
        UNIQUEIDENTIFIER, 
        ForeignKey('alerts.id', name='fk_comments_alerts'),
        nullable=True
    )
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'), nullable=True)
    content = Column(String(length=None), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    parent_id = Column(UNIQUEIDENTIFIER, ForeignKey('comments.id'), nullable=True)

    # Relationships
    alert = relationship("Alerts", back_populates="comments")
    user = relationship("Users", back_populates="comments")
    replies = relationship("Comments", 
                         backref=relationship("Comments", remote_side=[id]),
                         cascade="all, delete-orphan")

    def __init__(self, content: str, alert_id=None, user_id=None, parent_id=None):
        self.id = uuid4()
        self.content = content
        self.alert_id = alert_id
        self.user_id = user_id
        self.parent_id = parent_id

    def to_dict(self):
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_id": self.parent_id
        }