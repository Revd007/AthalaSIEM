from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from database.connection import Base
from sqlalchemy.sql import func
import uuid
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "siem"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User {self.username}>"

    @property
    def is_authenticated(self):
        return True

    @property
    def is_staff(self):
        return self.role in [UserRole.ADMIN, UserRole.ANALYST, UserRole.OPERATOR]

    @property
    def can_manage_users(self):
        return self.role == UserRole.ADMIN