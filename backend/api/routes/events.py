from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..schemas import EventCreate, EventResponse
from database.connection import get_db
from database.models import Event
from ai_engine.donquixote_service import DonquixoteService
from pathlib import Path

router = APIRouter()

# Use absolute path for config file
config_path = Path(__file__).parent.parent.parent / "ai_engine" / "services" / "ai_settings.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}")
    
ai_service = DonquixoteService(config_path)

@router.post("/events/", response_model=EventResponse)
async def create_event(event: EventCreate, db: Session = Depends(get_db)):
    db_event = Event(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

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

@router.post("/events/analyze", response_model=EventResponse)
async def analyze_event(event: EventCreate, db: Session = Depends(get_db)):
    # Analyze event using Donquixote Athala
    analysis = await ai_service.analyze_event(event.dict())
    
    if analysis['status'] == 'error':
        raise HTTPException(status_code=500, detail=analysis['error'])
    
    # Create event with AI analysis
    db_event = Event(
        **event.dict(),
        ai_analysis=analysis['analysis']
    )
    
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    
    return db_event