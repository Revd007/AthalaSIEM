from fastapi import APIRouter, Depends, Request, HTTPException, Query
from slowapi import Limiter
from slowapi.util import get_remote_address
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
async def create_alert(
    alert: AlertCreate, 
    db: Session = Depends(get_db)
):
    db_alert = Alert(**alert.dict())
    db.add(db_alert)
    await db.commit()
    await db.refresh(db_alert)
    
    # Trigger automated response if configured
    if alert.auto_respond:
        await playbook_engine.execute_playbook("default", {"alert": db_alert})
    
    return db_alert

@router.get("/alerts/", response_model=List[AlertResponse])
async def get_alerts(
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, le=100),
    db: Session = Depends(get_db)
):
    alerts = await db.query(Alert).offset(skip).limit(limit).all()
    return [
        {
            "id": str(alert.id),
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity,
            "status": alert.status,
            "created_at": alert.created_at,
            "updated_at": alert.updated_at
        } 
        for alert in alerts
    ]

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