from sqlalchemy import Column, String, DateTime, Boolean, Enum, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import enum
import uuid
from .base import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

# Association table
user_groups = Table(
    'user_groups',
    Base.metadata,
    Column('user_id', UNIQUEIDENTIFIER, ForeignKey('users.id'), primary_key=True),
    Column('group_id', UNIQUEIDENTIFIER, ForeignKey('groups.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(10), nullable=False, default='viewer')
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())

    # Relationships
    groups = relationship("Group", secondary=user_groups, back_populates="users")
    alerts = relationship("Alert", back_populates="assigned_to")
    audit_logs = relationship("AuditLog", back_populates="user")

    def __repr__(self):
        return f"<User {self.username}>"