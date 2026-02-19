"""
Threat intelligence: summary, indicators, check. Seed indicators so Total Indicators is non-zero.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_async_session
from db.tables import ThreatIntelIndicator

router = APIRouter(prefix="/api/threatintelligence", tags=["Threat Intelligence"])


@router.get("/summary")
async def get_summary(session: AsyncSession = Depends(get_async_session)):
    total = (await session.execute(select(func.count(ThreatIntelIndicator.id)))).scalar() or 0
    since = datetime.utcnow() - timedelta(days=1)
    r = await session.execute(
        select(ThreatIntelIndicator.source_feed, func.count(ThreatIntelIndicator.id))
        .where(ThreatIntelIndicator.last_seen >= since)
        .group_by(ThreatIntelIndicator.source_feed)
    )
    by_feed = [{"name": name or "General", "indicators": c, "matches": 0} for name, c in r.fetchall()]
    if not by_feed:
        by_feed = [{"name": "General", "indicators": total, "matches": 0}]
    return {
        "feeds": by_feed,
        "totalIndicators": total,
        "totalMatches": 0,
    }


@router.get("/indicators")
async def get_indicators(
    type: str | None = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session),
):
    q = select(ThreatIntelIndicator).order_by(ThreatIntelIndicator.last_seen.desc()).limit(min(limit, 500))
    if type:
        q = q.where(ThreatIntelIndicator.type == type)
    r = await session.execute(q)
    rows = r.scalars().all()
    return [
        {
            "id": i.id,
            "type": i.type,
            "value": i.value,
            "sourceFeed": i.source_feed,
            "confidence": i.confidence,
            "lastSeen": i.last_seen.isoformat() if i.last_seen else None,
        }
        for i in rows
    ]


class CheckRequest(BaseModel):
    value: str


@router.get("/correlations")
async def get_correlations(
    timeWindowHours: int = 24,
    minimumOccurrences: int = 3,
    session: AsyncSession = Depends(get_async_session),
):
    # Return real correlations from indicators seen in logs (simplified: group by value)
    since = datetime.utcnow() - timedelta(hours=min(timeWindowHours, 168))
    r = await session.execute(
        select(ThreatIntelIndicator.value, func.count(ThreatIntelIndicator.id))
        .where(ThreatIntelIndicator.last_seen >= since)
        .group_by(ThreatIntelIndicator.value)
        .having(func.count(ThreatIntelIndicator.id) >= minimumOccurrences)
    )
    rows = r.fetchall()
    return [
        {"pattern": val, "occurrences": c, "timeWindow": f"PT{timeWindowHours}H"}
        for val, c in rows
    ]


@router.post("/check")
async def check_ioc(body: CheckRequest, session: AsyncSession = Depends(get_async_session)):
    value = (body.value or "").strip()
    if not value:
        return {"matched": False, "indicators": []}
    r = await session.execute(select(ThreatIntelIndicator).where(ThreatIntelIndicator.value == value))
    indicators = r.scalars().all()
    return {
        "matched": len(indicators) > 0,
        "indicators": [
            {"type": i.type, "value": i.value, "sourceFeed": i.source_feed, "confidence": i.confidence}
            for i in indicators
        ],
    }
