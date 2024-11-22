from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, JSON, Enum, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
import uuid
import enum
from datetime import datetime
from .connection import Base

# Enum for user roles
class UserRole(enum.Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

# Association tables for many-to-many relationships
user_groups = Table(
    'user_groups',
    Base.metadata,
    Column('user_id', UNIQUEIDENTIFIER, ForeignKey('users.id')),
    Column('group_id', UNIQUEIDENTIFIER, ForeignKey('groups.id'))
)

class User(Base):
    __tablename__ = "users"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    groups = relationship("Group", secondary=user_groups, back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")
    alerts = relationship("Alert", back_populates="assigned_to")

class Group(Base):
    __tablename__ = "groups"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship("User", secondary=user_groups, back_populates="groups")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    severity = Column(Integer, nullable=False)  # 1-5 scale
    status = Column(String(50), nullable=False)
    source = Column(String(100))
    assigned_to_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    resolved_at = Column(DateTime)
    alert_metadata = Column(JSON)

    # Relationships
    assigned_to = relationship("User", back_populates="alerts")
    events = relationship("Event", back_populates="alert")

class Event(Base):
    __tablename__ = "events"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    alert_id = Column(UNIQUEIDENTIFIER, ForeignKey('alerts.id'))
    event_type = Column(String(50), nullable=False)
    source = Column(String(100))
    timestamp = Column(DateTime, nullable=False)
    severity = Column(Integer)
    message = Column(Text)
    raw_data = Column(JSON)
    processed_data = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    alert = relationship("Alert", back_populates="events")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    details = Column(JSON)
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")

class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON)
    description = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

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

class PlaybookRun(Base):
    __tablename__ = "playbook_runs"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    playbook_id = Column(String)
    status = Column(String)  # running, completed, failed, cancelled
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)

class PlaybookTemplate(Base):
    __tablename__ = "playbook_templates"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    content = Column(JSON, nullable=False)  # Stores the playbook definition/steps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)

class Tag(Base):
    __tablename__ = "tags"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    name = Column(String(50), nullable=False, unique=True)
    color = Column(String(7))  # Hex color code
    created_at = Column(DateTime, default=func.now())

class AlertTag(Base):
    __tablename__ = "alert_tags"

    alert_id = Column(Integer, ForeignKey('alerts.id'), primary_key=True)
    tag_id = Column(UNIQUEIDENTIFIER, ForeignKey('tags.id'), primary_key=True)
    added_at = Column(DateTime, default=func.now())
    added_by = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))

class Comment(Base):
    __tablename__ = "comments"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    alert_id = Column(Integer, ForeignKey('alerts.id'))
    user_id = Column(UNIQUEIDENTIFIER, ForeignKey('users.id'))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    parent_id = Column(UNIQUEIDENTIFIER, ForeignKey('comments.id'), nullable=True)