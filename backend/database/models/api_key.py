from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
from .base import Base

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    key = Column(String(255), unique=True, nullable=False)
    name = Column(String(100))
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    last_used_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey {self.name}>"