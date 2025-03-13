from typing import List, Dict, Any
from database.models import Alert
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlertService:
    async def get_alerts(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        try:
            query = db.query(Alert)
            
            if filters:
                if severity := filters.get('severity'):
                    query = query.filter(Alert.severity == severity)
                if status := filters.get('status'):
                    query = query.filter(Alert.status == status)
                if timeframe := filters.get('timeframe'):
                    hours = int(timeframe.replace('h', ''))
                    query = query.filter(
                        Alert.created_at >= datetime.utcnow() - timedelta(hours=hours)
                    )
                    
            return await query.offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            raise

    async def create_alert(self, db: Session, alert_data: Dict[str, Any]) -> Alert:
        """Create a new alert"""
        alert = Alert(**alert_data)
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        return alert

    async def update_alert(
        self,
        db: Session,
        alert_id: str,
        update_data: Dict[str, Any]
    ) -> Alert:
        """Update an existing alert"""
        alert = await db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            for key, value in update_data.items():
                setattr(alert, key, value)
            await db.commit()
            await db.refresh(alert)
        return alert 