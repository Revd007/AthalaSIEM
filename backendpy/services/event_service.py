from typing import List, Dict, Any
from database.models import Event
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EventService:
    async def get_events(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[Event]:
        """Get events with optional filtering"""
        try:
            query = db.query(Event)
            
            if filters:
                if event_type := filters.get('type'):
                    query = query.filter(Event.event_type == event_type)
                if severity := filters.get('severity'):
                    query = query.filter(Event.severity == severity)
                if source := filters.get('source'):
                    query = query.filter(Event.source == source)
                if timeframe := filters.get('timeframe'):
                    hours = int(timeframe.replace('h', ''))
                    query = query.filter(
                        Event.timestamp >= datetime.utcnow() - timedelta(hours=hours)
                    )
                    
            return await query.offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            raise

    async def create_event(self, db: Session, event_data: Dict[str, Any]) -> Event:
        """Create a new event"""
        event = Event(**event_data)
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event

    async def get_event_statistics(self, db: Session) -> Dict[str, Any]:
        """Get event statistics"""
        total = await db.query(Event).count()
        critical = await db.query(Event).filter(Event.severity == 'critical').count()
        high = await db.query(Event).filter(Event.severity == 'high').count()
        medium = await db.query(Event).filter(Event.severity == 'medium').count()
        low = await db.query(Event).filter(Event.severity == 'low').count()

        return {
            "total": total,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low
        } 