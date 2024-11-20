from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from ..schemas import AlertCreate, AlertResponse, AlertUpdate
from database.connection import get_db
from database.models import Alert
from automation.playbook_engine import PlaybookEngine

router = APIRouter()
playbook_engine = PlaybookEngine()

@router.post("/alerts/", response_model=AlertResponse)
async def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    db_alert = Alert(**alert.dict())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    
    # Trigger automated response if configured
    if alert.auto_respond:
        await playbook_engine.execute_playbook("default", {"alert": db_alert})
    
    return db_alert

@router.get("/alerts/", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    severity: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Alert)
    
    if severity:
        query = query.filter(Alert.severity == severity)
    if start_date:
        query = query.filter(Alert.timestamp >= start_date)
    if end_date:
        query = query.filter(Alert.timestamp <= end_date)
    
    alerts = query.offset(skip).limit(limit).all()
    return alerts

@router.put("/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db)
):
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not db_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    for key, value in alert_update.dict(exclude_unset=True).items():
        setattr(db_alert, key, value)
    
    db.commit()
    db.refresh(db_alert)
    return db_alert