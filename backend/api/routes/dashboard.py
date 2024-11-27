from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from auth.routes.auth import get_current_user
from database.connection import get_db
from auth.dependencies.auth_handler import AuthHandler
from database.models import Alert, Event as DBEvent, User
from sqlalchemy import func, select
from datetime import datetime, timedelta
from database.settings import settings
from typing import Dict, List, Any
from database.models.alert import Alert
import psutil
from ..models.dashboard import SystemMetrics, EventsOverview, Event
from ai_engine.donquixote_service import DonquixoteService

router = APIRouter()
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)

# Initialize DonquixoteService with default config
ai_service = DonquixoteService()

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    current_user: User = Depends(auth_handler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """Get dashboard statistics with AI insights"""
    try:
        # Get alerts count for last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        alerts_query = select(func.count()).select_from(Alert).where(
            Alert.created_at >= yesterday
        )
        alerts_result = await db.execute(alerts_query)
        alerts_count = alerts_result.scalar() or 0

        # Get events count
        events_query = select(func.count()).select_from(DBEvent)
        events_result = await db.execute(events_query)
        events_count = events_result.scalar() or 0

        # Get recent alerts with AI analysis
        recent_alerts_query = select(Alert).order_by(
            Alert.created_at.desc()
        ).limit(5)
        recent_alerts_result = await db.execute(recent_alerts_query)
        recent_alerts = recent_alerts_result.scalars().all()

        # Get AI service status
        ai_status = await ai_service.get_service_status()

        return {
            "alerts_24h": alerts_count,
            "total_events": events_count,
            "active_users": await get_active_users_count(db),
            "system_status": "healthy",
            "user_role": current_user.role,
            "last_updated": datetime.utcnow().isoformat(),
            "recent_alerts": [
                {
                    "id": str(alert.id),
                    "title": alert.title,
                    "severity": alert.severity,
                    "timestamp": alert.created_at.isoformat(),
                    "ai_analysis": await ai_service.analyze_event({
                        "type": "alert",
                        "severity": alert.severity,
                        "timestamp": alert.created_at.isoformat()
                    })
                }
                for alert in recent_alerts
            ],
            "ai_status": ai_status
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )

@router.get("/dashboard/recent-alerts")
async def get_recent_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(auth_handler.get_current_user)
) -> List[Dict]:
    """Get recent alerts with AI analysis"""
    try:
        query = select(Alert).order_by(Alert.created_at.desc()).limit(10)
        result = await db.execute(query)
        alerts = result.scalars().all()

        analyzed_alerts = []
        for alert in alerts:
            alert_data = {
                "id": str(alert.id),
                "title": alert.title,
                "severity": alert.severity,
                "timestamp": alert.created_at.isoformat()
            }
            
            # Add AI analysis
            ai_analysis = await ai_service.analyze_event({
                "type": "alert",
                "title": alert.title,
                "severity": alert.severity,
                "timestamp": alert.created_at.isoformat()
            })
            alert_data["ai_analysis"] = ai_analysis
            analyzed_alerts.append(alert_data)

        return analyzed_alerts
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching recent alerts: {str(e)}"
        )

@router.get("/system/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Get system metrics including AI resource usage"""
    try:
        # Get base system metrics
        base_metrics = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "storage": psutil.disk_usage('/').percent,
            "network": psutil.net_io_counters().bytes_sent / 1024 / 1024,
            "networkUsage": psutil.net_io_counters().bytes_recv / 1024 / 1024
        }

        # Get AI system specs
        ai_specs = ai_service._check_system_specs()

        return {
            **base_metrics,
            "ai_resources": ai_specs
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching system metrics: {str(e)}"
        )

@router.get("/events/overview", response_model=EventsOverview)
async def get_events_overview(
    current_user: User = Depends(auth_handler.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get events overview with AI analysis"""
    try:
        # Get events from database
        query = select(DBEvent).order_by(DBEvent.timestamp.desc()).limit(100)
        result = await db.execute(query)
        events = result.scalars().all()
        
        # Calculate metrics
        total = len(events)
        by_severity = {
            "critical": sum(1 for e in events if e.severity == "critical"),
            "warning": sum(1 for e in events if e.severity == "warning"),
            "normal": sum(1 for e in events if e.severity == "normal"),
            "high": sum(1 for e in events if e.severity == "high"),
            "medium": sum(1 for e in events if e.severity == "medium"),
            "low": sum(1 for e in events if e.severity == "low"),
            "info": sum(1 for e in events if e.severity == "info"),
            "error": sum(1 for e in events if e.severity == "error")
        }

        # Analyze recent events with AI
        recent_events = []
        for event in events[:20]:
            event_data = event.__dict__
            ai_analysis = await ai_service.analyze_event(event_data)
            recent_events.append({
                **event_data,
                "ai_analysis": ai_analysis
            })

        return {
            "total": total,
            "by_severity": by_severity,
            "recent_events": recent_events,
            "ai_insights": {
                "threat_patterns": await ai_service._identify_threat_chain(recent_events[0] if recent_events else {}),
                "risk_assessment": await ai_service._calculate_enhanced_risk(
                    0.5, 0.5,  # Default scores
                    ai_service._extract_temporal_features(recent_events[0] if recent_events else {}),
                    ai_service._extract_behavioral_features(recent_events[0] if recent_events else {})
                )
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching events overview: {str(e)}"
        )

@router.get("/events/recent", response_model=List[Event])
async def get_recent_events(
    limit: int = 20,
    current_user: User = Depends(auth_handler.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get recent events with AI analysis"""
    try:
        query = select(DBEvent).order_by(DBEvent.timestamp.desc()).limit(limit)
        result = await db.execute(query)
        events = result.scalars().all()

        analyzed_events = []
        for event in events:
            event_data = event.__dict__
            ai_analysis = await ai_service.analyze_event(event_data)
            analyzed_events.append({
                **event_data,
                "ai_analysis": ai_analysis
            })

        return analyzed_events
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching recent events: {str(e)}"
        )

@router.get("/ai/insights")
async def get_ai_insights() -> Dict[str, Any]:
    """Get comprehensive AI insights for dashboard"""
    try:
        # Get AI service status
        service_status = await ai_service.get_service_status()
        
        # Get system metrics
        system_metrics = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "storage": psutil.disk_usage('/').percent,
            "network": psutil.net_io_counters().bytes_sent / 1024 / 1024
        }

        return {
            "service_status": service_status,
            "system_metrics": system_metrics,
            "active_models": ai_service._get_active_models(),
            "knowledge_graph_status": {
                "nodes": len(ai_service.knowledge_graph.graph.nodes),
                "edges": len(ai_service.knowledge_graph.graph.edges)
            },
            "resource_usage": {
                "system": system_metrics,
                "ai_specific": ai_service._check_system_specs()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching AI insights: {str(e)}"
        )

async def get_active_users_count(db: AsyncSession) -> int:
    """Helper function to get active users count"""
    query = select(func.count()).select_from(User).where(User.is_active == True)
    result = await db.execute(query)
    return result.scalar() or 0