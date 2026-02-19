"""
Background worker: poll log_entries, run ML inference, write to ai_predictions and ai_anomalies.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session_factory
from db.tables import (
    LogEntry,
    ProcessedLogId,
    AIAnomaly,
    AIPrediction,
)
from ml.loader import get_model_service
from ml.feature_extractor import extract_features

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
POLL_INTERVAL_SEC = 30
LOOKBACK_HOURS = 24


async def _get_unprocessed_log_ids(session: AsyncSession) -> set[str]:
    result = await session.execute(select(ProcessedLogId.log_entry_id))
    return {r[0] for r in result.fetchall()}


async def _fetch_recent_logs(session: AsyncSession, exclude_ids: set[str], limit: int) -> list[LogEntry]:
    since = datetime.utcnow() - timedelta(hours=LOOKBACK_HOURS)
    q = select(LogEntry).where(LogEntry.Timestamp >= since).order_by(LogEntry.Timestamp.desc()).limit(limit)
    result = await session.execute(q)
    rows = result.scalars().all()
    return [r for r in rows if r.Id not in exclude_ids]


async def _process_batch(session: AsyncSession, logs: list[LogEntry]) -> None:
    if not logs:
        return
    model_svc = get_model_service()
    for log in logs:
        try:
            features = extract_features(
                message=log.Message or "",
                level=log.Level or "",
                source=log.Source or "",
                event_id=log.EventId or 0,
                category=log.Category,
            )
            pred_class, confidence = model_svc.predict_threat(features)
            anomaly_score, is_anomaly = model_svc.predict_anomaly(features)

            session.add(
                AIPrediction(
                    id=str(uuid4()),
                    log_entry_id=log.Id,
                    predicted_class=pred_class,
                    confidence=confidence,
                    explanation=None,
                    mitre_tactic=None,
                    mitre_technique=None,
                    model_name="random_forest",
                )
            )
            if is_anomaly or anomaly_score != 0.0:
                session.add(
                    AIAnomaly(
                        id=str(uuid4()),
                        log_entry_id=log.Id,
                        anomaly_score=anomaly_score,
                        anomaly_type="score",
                        severity="high" if anomaly_score > 0.7 else "medium",
                        model_name="gradient_boosting",
                    )
                )
            session.add(ProcessedLogId(id=str(uuid4()), log_entry_id=log.Id))
        except Exception as e:
            logger.warning("Inference failed for log %s: %s", log.Id, e)
    await session.commit()


async def run_log_analyzer_worker() -> None:
    """Run forever: poll logs, infer, write results."""
    while True:
        try:
            async with async_session_factory() as session:
                processed = await _get_unprocessed_log_ids(session)
                logs = await _fetch_recent_logs(session, processed, BATCH_SIZE)
                if logs:
                    await _process_batch(session, logs)
                    logger.info("Processed %d logs", len(logs))
        except asyncio.CancelledError:
            logger.info("Log analyzer worker cancelled")
            break
        except Exception as e:
            logger.exception("Log analyzer error: %s", e)
        await asyncio.sleep(POLL_INTERVAL_SEC)
