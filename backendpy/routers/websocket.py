"""
WebSocket endpoints for real-time anomaly and prediction push, and live hunt stream.
"""
import asyncio
from datetime import datetime, timedelta

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session_factory
from db.tables import AIAnomaly, AIPrediction, LogEntry

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/anomalies")
async def ws_anomalies(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            async with async_session_factory() as session:
                since = datetime.utcnow() - timedelta(minutes=15)
                r = await session.execute(
                    select(AIAnomaly).where(AIAnomaly.created_at >= since).order_by(AIAnomaly.created_at.desc()).limit(20)
                )
                items = r.scalars().all()
            await websocket.send_json({
                "type": "anomalies",
                "data": [
                    {"id": a.id, "logEntryId": a.log_entry_id, "score": a.anomaly_score, "severity": a.severity}
                    for a in items
                ],
                "ts": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/predictions")
async def ws_predictions(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            async with async_session_factory() as session:
                since = datetime.utcnow() - timedelta(minutes=15)
                r = await session.execute(
                    select(AIPrediction).where(AIPrediction.created_at >= since).order_by(AIPrediction.created_at.desc()).limit(20)
                )
                items = r.scalars().all()
            await websocket.send_json({
                "type": "predictions",
                "data": [
                    {"id": p.id, "predictedClass": p.predicted_class, "confidence": p.confidence}
                    for p in items
                ],
                "ts": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/live-hunt")
async def ws_live_hunt(websocket: WebSocket):
    await websocket.accept()
    try:
        query = await websocket.receive_text()
        q_lower = (query or "").lower().strip()
        async with async_session_factory() as session:
            stmt = select(LogEntry).order_by(LogEntry.Timestamp.desc()).limit(100)
            r = await session.execute(stmt)
            logs = r.scalars().all()
        for log in logs:
            text = (log.Message or "") + " " + (log.Source or "") + " " + (log.Level or "")
            if not q_lower or q_lower in text.lower():
                await websocket.send_json({
                    "id": log.Id,
                    "timestamp": log.Timestamp.isoformat() if log.Timestamp else None,
                    "message": (log.Message or "")[:200],
                    "source": log.Source,
                    "level": log.Level,
                })
            await asyncio.sleep(0.02)
        await websocket.send_json({"type": "done", "ts": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        pass
