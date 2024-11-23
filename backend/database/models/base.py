from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    username = Column(String(255), unique=True, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    role = Column(String(50), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(255))
    event_type = Column(String(255))
    severity = Column(Integer)
    message = Column(String(1000))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    host = Column(String(255))
    ip_address = Column(String(50))
    status = Column(String(50))
    raw_data = Column(String)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    description = Column(String(1000))
    severity = Column(Integer)
    status = Column(String(50))
    source = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())