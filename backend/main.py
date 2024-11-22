from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from api.routes import events, alerts, dashboard
from auth.routes.auth import router as auth_router
from collectors import windows_collector
from database.connection import AsyncSessionLocal, engine, get_db
from database.models import Base, Event, Alert, User as UserModel
import uvicorn
import asyncio
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from collectors.windows_collector import WindowsEventCollector
from collectors.network_collector import NetworkCollector
from ai_engine.correlation_engine import CorrelationEngine
from contextlib import asynccontextmanager
import sys
import ctypes
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.event import Event as EventSchema
from schemas.user import User as UserSchema

# Check if running as admin
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Verify admin privileges
if not is_admin():
    logging.warning("Application should be run as Administrator for full functionality")

# Create collector instance with configuration
collector_config = {
    'log_types': ['System', 'Security', 'Application'],
    'collection_interval': 10,  # seconds
    'enable_wmi': False  # Set to True only when running as Administrator
}

# Create collector instance
windows_collector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global windows_collector
    try:
        if is_admin():  # Only start collector if running as admin
            windows_collector = WindowsEventCollector(collector_config)
            # Start the event collection task
            asyncio.create_task(collect_events())
            # Start monitoring
            asyncio.create_task(windows_collector.monitor_system_changes())
        else:
            logging.error("Administrator privileges required for system monitoring")
    except Exception as e:
        logging.error(f"Error starting collector tasks: {e}")
    
    yield
    
    # Shutdown
    if windows_collector:
        windows_collector.cleanup()

app = FastAPI(lifespan=lifespan)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(events.router, prefix="/api/v1", tags=["events"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])

# Test Endpoints
@app.get("/test/health", tags=["test"])
async def health_check():
    """
    Test endpoint to check if the API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/test/events", response_model=List[Dict], tags=["test"])
async def test_events(db: Session = Depends(get_db)):
    """
    Test endpoint to get recent events
    """
    try:
        events = db.query(Event).order_by(Event.timestamp.desc()).limit(10).all()
        return [
            {
                "id": event.id,
                "timestamp": event.timestamp,
                "source": event.source,
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message
            } for event in events
        ]
    except Exception as e:
        logging.error(f"Error fetching test events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/alerts", response_model=List[Dict], tags=["test"])
async def test_alerts(db: Session = Depends(get_db)):
    """
    Test endpoint to get recent alerts
    """
    try:
        alerts = db.query(Alert).order_by(Alert.timestamp.desc()).limit(10).all()
        return [
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "title": alert.title,
                "severity": alert.severity,
                "status": alert.status
            } for alert in alerts
        ]
    except Exception as e:
        logging.error(f"Error fetching test alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/collectors/windows", tags=["test"])
async def test_windows_collector():
    """
    Test endpoint to check Windows event collection
    """
    try:
        collector = WindowsEventCollector({
            'log_types': ['System', 'Security'],
            'collection_interval': 10
        })
        
        events = []
        async for event in collector.collect_logs():
            events.append(event)
            if len(events) >= 5:  # Collect 5 events
                break
                
        return {
            "status": "success",
            "collected_events": events,
            "count": len(events)
        }
    except Exception as e:
        logging.error(f"Error testing Windows collector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/collectors/network", tags=["test"])
async def test_network_collector():
    """
    Test endpoint to check network event collection
    """
    try:
        collector = NetworkCollector()
        events = []
        async for event in collector.start_collection():
            events.append(event)
            if len(events) >= 5:  # Collect 5 events
                break
                
        return {
            "status": "success",
            "collected_events": events,
            "count": len(events)
        }
    except Exception as e:
        logging.error(f"Error testing network collector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/correlation", tags=["test"])
async def test_correlation_engine():
    """
    Test endpoint to check correlation engine
    """
    try:
        correlation_engine = CorrelationEngine()
        test_event = {
            "timestamp": datetime.utcnow(),
            "source": "test",
            "event_type": "login_failure",
            "severity": 1,
            "message": "Failed login attempt",
            "user": "test_user",
            "ip_address": "192.168.1.100"
        }
        
        result = await correlation_engine.process_event(test_event)
        return {
            "status": "success",
            "test_event": test_event,
            "correlation_result": result
        }
    except Exception as e:
        logging.error(f"Error testing correlation engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def collect_events():
    """Background task to collect events"""
    if not windows_collector:
        return
        
    try:
        async for event in windows_collector.collect_logs():
            if event:
                logging.info(f"Collected event: {event.get('event_id')} from {event.get('source')}")
                try:
                    # Process with AI
                    ai_result = await windows_collector.correlation_engine.process_event(event)
                    
                    # Use async database session
                    async with AsyncSessionLocal() as session:
                        try:
                            # Create new event
                            new_event = Event(
                                timestamp=event['timestamp'],
                                source=event['source'],
                                event_type=event['event_type'],
                                severity=event['severity'],
                                message=event['message'],
                                ai_analysis=ai_result
                            )
                            
                            # Add and commit asynchronously
                            session.add(new_event)
                            await session.commit()
                            
                        except Exception as db_error:
                            await session.rollback()
                            logging.error(f"Database error: {db_error}")
                            
                except Exception as e:
                    logging.error(f"Error processing event: {e}")
                    
    except Exception as e:
        logging.error(f"Error in collect_events: {e}")

@app.get("/events/", response_model=List[EventSchema])
async def read_events(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Get list of events"""
    try:
        async with db as session:
            result = await session.execute(
                select(Event).offset(skip).limit(limit)
            )
            events = result.scalars().all()
            return events
    except Exception as e:
        logging.error(f"Error reading events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)