from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime, timedelta
from ...database.connection import get_db
from ...database.models import Event, Alert
from sqlalchemy import func

router = APIRouter()

@router.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    
    # Get event counts by severity
    event_counts = db.query(
        Event.severity,
        func.count(Event.id)
    ).filter(
        Event.timestamp >= last_24h
    ).group_by(Event.severity).all()
    
    # Get alert counts by status
    alert_counts = db.query(
        Alert.status,
        func.count(Alert.id)
    ).filter(
        Alert.timestamp >= last_24h
    ).group_by(Alert.status).all()
    
    return {
        "events": {severity: count for severity, count in event_counts},
        "alerts": {status: count for status, count in alert_counts},
        "last_updated": now
    }

@router.get("/dashboard/top-sources")
async def get_top_sources(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    last_24h = datetime.utcnow() - timedelta(hours=24)
    
    top_sources = db.query(
        Event.source,
        func.count(Event.id).label('count')
    ).filter(
        Event.timestamp >= last_24h
    ).group_by(Event.source).order_by(
        func.count(Event.id).desc()
    ).limit(limit).all()
    
    return [{"source": source, "count": count} for source, count in top_sources]

@router.get("/dashboard/threat-timeline")
async def get_threat_timeline(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    timeline = db.query(
        func.date_trunc('hour', Alert.timestamp).label('hour'),
        func.count(Alert.id).label('count')
    ).filter(
        Alert.timestamp >= start_time
    ).group_by(
        func.date_trunc('hour', Alert.timestamp)
    ).order_by(
        func.date_trunc('hour', Alert.timestamp)
    ).all()
    
    return [{"hour": hour, "count": count} for hour, count in timeline]