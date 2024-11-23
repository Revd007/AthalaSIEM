from sqlalchemy import Column, String, DateTime, Boolean, func, Enum
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from .base import Base
import uuid
import enum

# Define UserRole enum
class UserRole(str, enum.Enum):
    ADMIN = 'admin'
    ANALYST = 'analyst'
    OPERATOR = 'operator'
    VIEWER = 'viewer'

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "siem"}

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.getdate())
    updated_at = Column(DateTime, default=func.getdate())