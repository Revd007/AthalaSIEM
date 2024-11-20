from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String)
    event_type = Column(String)
    severity = Column(Integer)
    message = Column(String)
    raw_data = Column(JSON)
    host = Column(String)
    ip_address = Column(String)
    status = Column(String)
    alert_id = Column(Integer, ForeignKey('alerts.id'), nullable=True)

class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    title = Column(String)
    description = Column(String)
    severity = Column(Integer)
    status = Column(String)
    source = Column(String)
    events = relationship('Event', backref='alert')
    playbook_runs = relationship('PlaybookRun', backref='alert')

class PlaybookRun(Base):
    __tablename__ = 'playbook_runs'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey('alerts.id'))
    playbook_id = Column(String)
    status = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    result = Column(JSON)

class AnomalyScore(Base):
    __tablename__ = 'anomaly_scores'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String)
    score = Column(Float)
    threshold = Column(Float)
    is_anomaly = Column(Boolean)