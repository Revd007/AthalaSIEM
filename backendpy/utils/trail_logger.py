from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request
from sqlalchemy.orm import Session
from backendpy.models import TrailLog

class TrailLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def log_action(
        self,
        db: Session,
        user_id: str,
        action: str,
        component: str,
        details: Dict[str, Any],
        request: Optional[Request] = None
    ) -> None:
        """
        Log an action to the trail log system
        """
        try:
            # Create trail log entry
            trail_log = TrailLog(
                user_id=user_id,
                action=action,
                component=component,
                details=json.dumps(details),
                timestamp=datetime.utcnow(),
                user_agent=request.headers.get("user-agent", "") if request else "",
                ip_address=request.client.host if request else ""
            )

            # Save to database
            db.add(trail_log)
            db.commit()

            # Also log to file
            self.logger.info(
                f"Trail Log - User: {user_id}, Action: {action}, "
                f"Component: {component}, Details: {details}"
            )

        except Exception as e:
            self.logger.error(f"Error logging trail: {str(e)}")
            db.rollback()

    def get_user_actions(
        self,
        db: Session,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list:
        """
        Get trail logs for a specific user
        """
        query = db.query(TrailLog).filter(TrailLog.user_id == user_id)

        if start_date:
            query = query.filter(TrailLog.timestamp >= start_date)
        if end_date:
            query = query.filter(TrailLog.timestamp <= end_date)

        return query.order_by(TrailLog.timestamp.desc()).all()

    def get_action_stats(
        self,
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about trail logs
        """
        query = db.query(TrailLog)

        if start_date:
            query = query.filter(TrailLog.timestamp >= start_date)
        if end_date:
            query = query.filter(TrailLog.timestamp <= end_date)

        total_logs = query.count()
        actions_by_user = (
            db.query(TrailLog.user_id, db.func.count(TrailLog.id))
            .group_by(TrailLog.user_id)
            .all()
        )
        actions_by_type = (
            db.query(TrailLog.action, db.func.count(TrailLog.id))
            .group_by(TrailLog.action)
            .all()
        )

        return {
            "total_logs": total_logs,
            "actions_by_user": dict(actions_by_user),
            "actions_by_type": dict(actions_by_type)
        }

# Create singleton instance
trail_logger = TrailLogger() 