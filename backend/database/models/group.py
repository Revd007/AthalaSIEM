from sqlalchemy import Column, String, DateTime, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
from .base import Base

class Group(Base):
    __tablename__ = "groups"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())

    # Relationships
    users = relationship("User", secondary="user_groups", back_populates="groups")

    def __repr__(self):
        return f"<Group {self.name}>"