from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER, NTEXT
from .base import Base
import uuid

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = {"schema": "dbo"}

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(NTEXT)
    severity = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)
    source = Column(String(100))
    assigned_to_id = Column(UNIQUEIDENTIFIER, ForeignKey('dbo.users.id'))
    created_at = Column(DateTime, default=func.getdate())
    updated_at = Column(DateTime, default=func.getdate())
    resolved_at = Column(DateTime)
    alert_metadata = Column(String)