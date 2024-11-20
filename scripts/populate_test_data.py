import asyncio
from backend.database.models import Event, Alert
from backend.database.connection import get_db
from datetime import datetime

async def populate_test_data():
    db = await anext(get_db())
    
    # Create sample events
    test_events = [
        Event(
            timestamp=datetime.utcnow(),
            source="Windows Security",
            event_type="Login",
            severity=1,
            message="Failed login attempt",
            user="test_user",
            computer="TEST-PC",
            raw_data={"attempt_count": 3}
        ),
        Event(
            timestamp=datetime.utcnow(),
            source="Firewall",
            event_type="Connection",
            severity=2,
            message="Suspicious outbound connection",
            ip_address="192.168.1.100",
            raw_data={"port": 445}
        )
    ]
    
    db.add_all(test_events)
    await db.commit()
    
    # Create sample alert
    test_alert = Alert(
        timestamp=datetime.utcnow(),
        title="Suspicious Activity Detected",
        description="Multiple failed login attempts",
        severity=1,
        status="open",
        source="IDS"
    )
    
    db.add(test_alert)
    await db.commit()

if __name__ == "__main__":
    asyncio.run(populate_test_data())