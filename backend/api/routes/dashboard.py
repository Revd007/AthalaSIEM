from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database.connection import get_db
from auth.dependencies.auth_handler import AuthHandler
from database.models import Alert, Event, User
from sqlalchemy import func, select
from datetime import datetime, timedelta
from database.settings import settings
from typing import Dict, List
from database.models.alert import Alert
from database.models.event import Event

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
        events_query = select(func.count()).select_from(Event)
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