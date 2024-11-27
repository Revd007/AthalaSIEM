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
from ..services.ai_service import analyze_events
from ai_engine.services.ai_service_manager import AIServiceManager

router = APIRouter()
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    current_user: User = Depends(auth_handler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """Get dashboard statistics"""
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

        # Get recent alerts
        recent_alerts_query = select(Alert).order_by(
            Alert.created_at.desc()
        ).limit(5)
        recent_alerts_result = await db.execute(recent_alerts_query)
        recent_alerts = recent_alerts_result.scalars().all()

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
                    "timestamp": alert.created_at.isoformat()
                }
                for alert in recent_alerts
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )

async def get_active_users_count(db: AsyncSession) -> int:
    query = select(func.count()).select_from(User).where(User.is_active == True)
    result = await db.execute(query)
    return result.scalar() or 0

@router.get("/dashboard/recent-alerts")
async def get_recent_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(auth_handler.get_current_user)
) -> List[Dict]:
    """Get recent alerts for dashboard"""
    try:
        query = select(Alert).order_by(Alert.created_at.desc()).limit(10)
        result = await db.execute(query)
        alerts = result.scalars().all()

        return [
            {
                "id": str(alert.id),
                "title": alert.title,
                "severity": alert.severity,
                "timestamp": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching recent alerts: {str(e)}"
        )

@router.get("/system/metrics", response_model=SystemMetrics)
async def get_system_metrics(current_user = Depends(get_current_user)):
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "storage": psutil.disk_usage('/').percent,
        "network": psutil.net_io_counters().bytes_sent / 1024 / 1024,  # Convert to MB
        "networkUsage": psutil.net_io_counters().bytes_recv / 1024 / 1024  # Convert to MB
    }

@router.get("/events/overview", response_model=EventsOverview)
async def get_events_overview(
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    # Get events from database
    events = db.query(DBEvent).order_by(DBEvent.timestamp.desc()).limit(100).all()
    
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
    
    # Prepare chart data
    chart_data = prepare_chart_data(events)
    
    # Get recent events with AI analysis
    recent_events = events[:20]
    for event in recent_events:
        if not event.ai_analysis:
            event.ai_analysis = analyze_events(event)
            db.commit()
    
    return {
        "total": total,
        "by_severity": by_severity,
        "chart_data": chart_data,
        "recent_events": recent_events
    }

@router.get("/events/recent", response_model=List[Event])
async def get_recent_events(
    limit: int = 20,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    events = db.query(DBEvent).order_by(DBEvent.timestamp.desc()).limit(limit).all()
    return events

@router.get("/ai/insights")
async def get_ai_insights() -> Dict[str, Any]:
    """Get AI insights for dashboard"""
    try:
        ai_manager = AIServiceManager({
            'ai_enabled': True,
            'resource_settings': {
                'max_memory_usage': 1024,  # MB
                'max_cpu_usage': 80,  # percent
                'max_storage_usage': 90,  # percent
                'max_network_usage': 1000  # MB/s
            },
            'feature_toggles': {
                'anomaly_detection': True,
                'threat_analysis': True
            }
        })
        
        insights = await ai_manager.get_ai_status()
        return {
            "insights": [
                {
                    "title": "System Health",
                    "description": f"Memory: {insights['resource_usage']['memory_used']}%, CPU: {insights['resource_usage']['cpu_used']}%, Storage: {insights['resource_usage']['storage_used']}%, Network: {insights['resource_usage']['network_used']} MB/s"
                },
                {
                    "title": "AI System Status", 
                    "description": f"AI System is {'running' if insights['is_running'] else 'disabled'}"
                }
            ],
            "status": insights
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching AI insights: {str(e)}"
        )