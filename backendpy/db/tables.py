"""
SQLAlchemy models for AthalaSIEM Python backend.
- LogEntry: read-only mirror of .NET log_entries table.
- All ai_* and supporting tables: created and owned by Python.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ----- Read-only mirror of .NET log_entries (table: log_entries) -----
class LogEntry(Base):
    __tablename__ = "log_entries"
    __table_args__ = {"info": {"read_only": True}}

    Id: Mapped[str] = mapped_column("Id", String(36), primary_key=True)
    AgentId: Mapped[str] = mapped_column("AgentId", String(36), nullable=False)
    Timestamp: Mapped[datetime] = mapped_column("Timestamp", DateTime(timezone=True), nullable=False)
    Level: Mapped[str] = mapped_column("Level", String(50), nullable=False)
    Message: Mapped[str] = mapped_column("Message", Text, nullable=False)
    Source: Mapped[str] = mapped_column("Source", String(255), nullable=False)
    Category: Mapped[Optional[str]] = mapped_column("Category", String(100), nullable=True)
    EventId: Mapped[int] = mapped_column("EventId", Integer, default=0)
    IPAddress: Mapped[str] = mapped_column("IPAddress", String(45), default="")
    Exception: Mapped[Optional[str]] = mapped_column("Exception", Text, nullable=True)
    MachineName: Mapped[str] = mapped_column("MachineName", String(255), default="")
    ProcessId: Mapped[int] = mapped_column("ProcessId", Integer, default=0)
    ThreadId: Mapped[int] = mapped_column("ThreadId", Integer, default=0)
    UserId: Mapped[Optional[str]] = mapped_column("UserId", String(255), nullable=True)
    RequestPath: Mapped[Optional[str]] = mapped_column("RequestPath", String(500), nullable=True)
    RequestId: Mapped[Optional[str]] = mapped_column("RequestId", String(50), nullable=True)
    ClientIp: Mapped[Optional[str]] = mapped_column("ClientIp", String(45), nullable=True)
    Properties: Mapped[Optional[str]] = mapped_column("Properties", Text, nullable=True)
    ReceivedAt: Mapped[datetime] = mapped_column("ReceivedAt", DateTime(timezone=True), nullable=False)
    Processed: Mapped[bool] = mapped_column("Processed", Boolean, default=False)
    ProcessedAt: Mapped[Optional[datetime]] = mapped_column("ProcessedAt", DateTime(timezone=True), nullable=True)
    CreatedAt: Mapped[datetime] = mapped_column("CreatedAt", DateTime(timezone=True), nullable=False)
    StackTrace: Mapped[Optional[str]] = mapped_column("StackTrace", Text, nullable=True)
    Details: Mapped[Optional[str]] = mapped_column("Details", Text, nullable=True)


# ----- Python-owned AI tables (create if not exist) -----
class ProcessedLogId(Base):
    """Tracks which log IDs we have already run through the ML pipeline."""
    __tablename__ = "processed_log_ids"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_entry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class AIAnomaly(Base):
    __tablename__ = "ai_anomalies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_entry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(100), default="")
    severity: Mapped[str] = mapped_column(String(50), default="medium")
    model_name: Mapped[str] = mapped_column(String(100), default="gradient_boosting")
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class AIPrediction(Base):
    __tablename__ = "ai_predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_entry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    predicted_class: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mitre_tactic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mitre_technique: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), default="random_forest")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class AIHunt(Base):
    __tablename__ = "ai_hunts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="running")
    findings_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    findings: Mapped[list["AIHuntFinding"]] = relationship("AIHuntFinding", back_populates="hunt")


class AIHuntFinding(Base):
    __tablename__ = "ai_hunt_findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    hunt_id: Mapped[str] = mapped_column(String(36), ForeignKey("ai_hunts.id"), nullable=False)
    log_entry_id: Mapped[str] = mapped_column(String(36), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(50), default="medium")
    matched_rule: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    hunt: Mapped["AIHunt"] = relationship("AIHunt", back_populates="findings")


class DetectionRule(Base):
    __tablename__ = "detection_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type: Mapped[str] = mapped_column(String(20), nullable=False)  # yara | sigma
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(50), default="medium")
    status: Mapped[str] = mapped_column(String(50), default="active")
    matches_count: Mapped[int] = mapped_column(Integer, default=0)
    last_tested: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class ThreatIntelIndicator(Base):
    __tablename__ = "threat_intel_indicators"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # ip | domain | hash | url | email
    value: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    source_feed: Mapped[str] = mapped_column(String(255), default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class PlaybookDefinition(Base):
    __tablename__ = "playbook_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(100), default="")
    steps_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class PlaybookExecution(Base):
    __tablename__ = "playbook_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    playbook_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="running")
    results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
