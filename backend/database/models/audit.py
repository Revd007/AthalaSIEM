from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
from .base import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    details = Column(String)  # NVARCHAR(MAX) for JSON
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} by {self.user_id}>"