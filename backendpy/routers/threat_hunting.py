"""
Threat Hunting: dashboard, IOC scan, behavior (MITRE) from ai_predictions.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_async_session
from db.tables import AIHunt, AIHuntFinding, AIPrediction, ThreatIntelIndicator

router = APIRouter(prefix="/api/threat-hunting", tags=["Threat Hunting"])


@router.get("/dashboard")
async def get_dashboard(session: AsyncSession = Depends(get_async_session)):
    since = datetime.utcnow() - timedelta(days=7)
    hunts = await session.execute(select(AIHunt).where(AIHunt.created_at >= since).order_by(AIHunt.started_at.desc()))
    hunt_list = hunts.scalars().all()
    total_findings = sum(h.findings_count for h in hunt_list)
    completed = [h for h in hunt_list if h.status == "completed"]
    recent_findings = []
    for h in hunt_list[:5]:
        r2 = await session.execute(select(AIHuntFinding).where(AIHuntFinding.hunt_id == h.id).order_by(AIHuntFinding.created_at.desc()).limit(3))
        for f in r2.scalars().all():
            recent_findings.append({
                "id": f.id,
                "huntId": f.hunt_id,
                "description": f.description,
                "severity": f.severity,
                "createdAt": f.created_at.isoformat() if f.created_at else None,
            })
    recent_findings = recent_findings[:15]

    return {
        "huntActivityLast7Days": [
            {"date": h.started_at.date().isoformat() if h.started_at else None, "count": h.findings_count}
            for h in hunt_list
        ],
        "activeHunts": len([h for h in hunt_list if h.status == "running"]),
        "totalFindings": total_findings,
        "avgHuntDuration": 0,
        "successRate": len(completed) / len(hunt_list) * 100 if hunt_list else 0,
        "recentFindings": recent_findings,
    }


class IOCScanRequest(BaseModel):
    value: str
    types: list[str] | None = None  # ip, domain, hash, url, email


@router.post("/ioc/scan")
async def scan_ioc(body: IOCScanRequest, session: AsyncSession = Depends(get_async_session)):
    value = (body.value or "").strip()
    if not value:
        return {"matchesFound": 0, "results": [], "historicalMatches": []}
    r = await session.execute(
        select(ThreatIntelIndicator).where(ThreatIntelIndicator.value == value)
    )
    indicators = r.scalars().all()
    return {
        "matchesFound": len(indicators),
        "results": [
            {
                "type": i.type,
                "value": i.value,
                "sourceFeed": i.source_feed,
                "confidence": i.confidence,
            }
            for i in indicators
        ],
        "historicalMatches": [],
    }


class LiveHuntStartRequest(BaseModel):
    query: str
    timeRangeMinutes: int = 15


@router.post("/live/start")
async def live_hunt_start(body: LiveHuntStartRequest, session: AsyncSession = Depends(get_async_session)):
    from datetime import timedelta
    from uuid import uuid4
    since = datetime.utcnow() - timedelta(minutes=body.timeRangeMinutes)
    hunt = AIHunt(
        id=str(uuid4()),
        name="Live hunt",
        query=body.query,
        status="running",
        findings_count=0,
    )
    session.add(hunt)
    await session.commit()
    from sqlalchemy import select
    from db.tables import LogEntry
    q_lower = (body.query or "").lower()
    r = await session.execute(select(LogEntry).where(LogEntry.Timestamp >= since).order_by(LogEntry.Timestamp.desc()).limit(500))
    logs = r.scalars().all()
    count = 0
    for log in logs:
        text = (log.Message or "") + " " + (log.Source or "")
        if q_lower in text.lower():
            session.add(AIHuntFinding(id=str(uuid4()), hunt_id=hunt.id, log_entry_id=log.Id, description=(log.Message or "")[:100], severity="medium"))
            count += 1
    hunt.status = "completed"
    hunt.findings_count = count
    hunt.completed_at = datetime.utcnow()
    await session.commit()
    return {"sessionId": hunt.id, "findingsCount": count, "status": "completed"}


@router.get("/live/{session_id}/results")
async def live_hunt_results(session_id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(AIHunt).where(AIHunt.id == session_id))
    hunt = r.scalar_one_or_none()
    if not hunt:
        raise HTTPException(404, "Hunt not found")
    r2 = await session.execute(select(AIHuntFinding).where(AIHuntFinding.hunt_id == session_id))
    findings = r2.scalars().all()
    return {
        "sessionId": session_id,
        "status": hunt.status,
        "findingsCount": hunt.findings_count,
        "findings": [{"id": f.id, "logEntryId": f.log_entry_id, "description": f.description, "severity": f.severity} for f in findings],
    }


@router.get("/behavior")
async def get_behavior(session: AsyncSession = Depends(get_async_session)):
    since = datetime.utcnow() - timedelta(hours=24)
    # MITRE technique counts from predictions
    r = await session.execute(
        select(AIPrediction.mitre_technique, func.count(AIPrediction.id))
        .where(AIPrediction.created_at >= since, AIPrediction.mitre_technique.isnot(None))
        .group_by(AIPrediction.mitre_technique)
    )
    rows = r.fetchall()
    technique_counts = [{"technique": t or "Unknown", "count": c} for t, c in rows]
    return {
        "mitreTechniqueCounts": technique_counts,
        "processBehavior": [],
        "networkBehavior": [],
        "userBehavior": [],
    }
