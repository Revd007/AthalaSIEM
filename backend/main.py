from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from pytest import Session
from api.routes import events, alerts, dashboard
from auth.routes.auth import router as auth_router
from database.connection import engine, get_db
from database.models import Base, Event, Alert
import uvicorn
import asyncio
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from collectors.windows_collector import WindowsEventCollector
from collectors.network_collector import NetworkCollector
from ai_engine.correlation_engine import CorrelationEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/siem.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="SIEM Solution API")

# CORS configuration
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

@app.on_event("startup")
async def startup_event():
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize collectors
    windows_collector = WindowsEventCollector({
        'log_types': ['System', 'Security'],
        'collection_interval': 10
    })
    network_collector = NetworkCollector()
    
    # Initialize correlation engine
    correlation_engine = CorrelationEngine()
    
    # Start collection tasks
    asyncio.create_task(start_collectors(
        windows_collector,
        network_collector,
        correlation_engine
    ))

async def start_collectors(windows_collector, network_collector, correlation_engine):
    while True:
        try:
            # Collect and process events
            windows_events = windows_collector.collect_logs()
            network_events = network_collector.start_collection()
            
            async for event in windows_events:
                await correlation_engine.process_event(event)
            
            async for event in network_events:
                await correlation_engine.process_event(event)
                
        except Exception as e:
            logging.error(f"Error in collectors: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)