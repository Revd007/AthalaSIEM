from asyncio import Event
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pathlib import Path
import yaml

from database.connection import get_db
from ..schemas import EventCreate, EventResponse
from ai_engine.donquixote_service import DonquixoteService

router = APIRouter()

# Initialize AI service
config_path = Path("backend/ai_engine/services/ai_settings.yaml")
try:
    if config_path.exists():
        with open(config_path) as f:
            ai_config = yaml.safe_load(f)
            ai_service = DonquixoteService(config=ai_config)
    else:
        # Use default configuration
        ai_service = DonquixoteService()
except Exception as e:
    print(f"Error initializing AI service: {e}")
    ai_service = DonquixoteService()  # Fallback to default config

@router.post("/events/", response_model=EventResponse)
async def create_event(event: EventCreate, db: Session = Depends(get_db)):
    try:
        # Analyze event with AI
        if ai_service:
            analysis_result = await ai_service.analyze_event(event.dict())
            event.ai_analysis = analysis_result
        
        # Create event in database
        db_event = Event(**event.dict())
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        return db_event
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/", response_model=List[EventResponse])
async def get_events(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    events = db.query(Event).offset(skip).limit(limit).all()
    return events

@router.get("/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event