from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from database.connection import Base
from sqlalchemy.sql import func
import uuid
from database.enums import UserRole

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "dbo"}

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(50), nullable=False, default=UserRole.VIEWER.value)
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
        return self.role in [UserRole.ADMIN.value, UserRole.ANALYST.value, UserRole.OPERATOR.value]

    @property
    def can_manage_users(self):
        return self.role == UserRole.ADMIN.value