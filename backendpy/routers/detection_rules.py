"""
YARA and Sigma detection rules: CRUD + test against recent logs.
"""
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_async_session
from db.tables import DetectionRule, LogEntry

router = APIRouter(prefix="/api/detection-rules", tags=["Detection Rules"])


class RuleCreate(BaseModel):
    name: str
    content: str
    severity: str = "medium"
    status: str = "active"


class RuleUpdate(BaseModel):
    name: str | None = None
    content: str | None = None
    severity: str | None = None
    status: str | None = None


# ----- YARA -----
@router.get("/yara")
async def list_yara(session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(
        select(DetectionRule).where(DetectionRule.type == "yara").order_by(DetectionRule.updated_at.desc())
    )
    rules = r.scalars().all()
    return [
        {
            "id": x.id,
            "name": x.name,
            "description": "",
            "severity": x.severity,
            "status": x.status,
            "matches": x.matches_count,
            "content": x.content,
            "lastModified": x.updated_at.isoformat() if x.updated_at else None,
        }
        for x in rules
    ]


@router.get("/yara/{id}")
async def get_yara(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "yara"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    return {
        "id": rule.id,
        "name": rule.name,
        "description": "",
        "severity": rule.severity,
        "status": rule.status,
        "matches": rule.matches_count,
        "content": rule.content,
        "lastModified": rule.updated_at.isoformat() if rule.updated_at else None,
    }


@router.post("/yara", status_code=201)
async def create_yara(body: RuleCreate, session: AsyncSession = Depends(get_async_session)):
    rule = DetectionRule(
        id=str(uuid4()),
        type="yara",
        name=body.name,
        content=body.content,
        severity=body.severity,
        status=body.status,
    )
    session.add(rule)
    await session.commit()
    return {"id": rule.id, "name": rule.name, "severity": rule.severity, "status": rule.status, "content": rule.content}


@router.put("/yara/{id}")
async def update_yara(id: str, body: RuleUpdate, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "yara"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    if body.name is not None:
        rule.name = body.name
    if body.content is not None:
        rule.content = body.content
    if body.severity is not None:
        rule.severity = body.severity
    if body.status is not None:
        rule.status = body.status
    rule.updated_at = datetime.utcnow()
    await session.commit()
    return {"id": rule.id, "name": rule.name}


@router.delete("/yara/{id}", status_code=204)
async def delete_yara(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "yara"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    await session.delete(rule)
    await session.commit()


@router.post("/yara/{id}/test")
async def test_yara(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "yara"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    # Run against recent log messages (content as text)
    logs = await session.execute(select(LogEntry).order_by(LogEntry.Timestamp.desc()).limit(200))
    rows = logs.scalars().all()
    import time
    start = time.perf_counter()
    matches = 0
    for log in rows:
        text = (log.Message or "") + " " + (log.Source or "")
        if rule.content and rule.content.strip().lower() in text.lower():
            matches += 1
    elapsed = time.perf_counter() - start
    rule.last_tested = datetime.utcnow()
    rule.matches_count = matches
    await session.commit()
    return {"ruleId": id, "success": True, "matches": matches, "executionTime": round(elapsed, 3), "testedAt": datetime.utcnow().isoformat()}


# ----- Sigma -----
@router.get("/sigma")
async def list_sigma(session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(
        select(DetectionRule).where(DetectionRule.type == "sigma").order_by(DetectionRule.updated_at.desc())
    )
    rules = r.scalars().all()
    return [
        {
            "id": x.id,
            "title": x.name,
            "description": "",
            "level": x.severity,
            "logsource": "",
            "tags": [],
            "matches": x.matches_count,
            "content": x.content,
            "lastModified": x.updated_at.isoformat() if x.updated_at else None,
        }
        for x in rules
    ]


@router.get("/sigma/{id}")
async def get_sigma(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "sigma"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    return {
        "id": rule.id,
        "title": rule.name,
        "description": "",
        "level": rule.severity,
        "logsource": "",
        "tags": [],
        "matches": rule.matches_count,
        "content": rule.content,
        "lastModified": rule.updated_at.isoformat() if rule.updated_at else None,
    }


@router.post("/sigma", status_code=201)
async def create_sigma(body: RuleCreate, session: AsyncSession = Depends(get_async_session)):
    rule = DetectionRule(
        id=str(uuid4()),
        type="sigma",
        name=body.name,
        content=body.content,
        severity=body.severity,
        status=body.status,
    )
    session.add(rule)
    await session.commit()
    return {"id": rule.id, "title": rule.name, "level": rule.severity, "content": rule.content}


@router.put("/sigma/{id}")
async def update_sigma(id: str, body: RuleUpdate, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "sigma"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    if body.name is not None:
        rule.name = body.name
    if body.content is not None:
        rule.content = body.content
    if body.severity is not None:
        rule.severity = body.severity
    if body.status is not None:
        rule.status = body.status
    rule.updated_at = datetime.utcnow()
    await session.commit()
    return {"id": rule.id, "title": rule.name}


@router.delete("/sigma/{id}", status_code=204)
async def delete_sigma(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "sigma"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    await session.delete(rule)
    await session.commit()


@router.post("/sigma/{id}/test")
async def test_sigma(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(DetectionRule).where(DetectionRule.id == id, DetectionRule.type == "sigma"))
    rule = r.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")
    # Simple keyword match against log message (full Sigma-to-SQL would require a parser)
    logs = await session.execute(select(LogEntry).order_by(LogEntry.Timestamp.desc()).limit(200))
    rows = logs.scalars().all()
    import time
    start = time.perf_counter()
    matches = 0
    for log in rows:
        text = (log.Message or "") + " " + (log.Source or "")
        if rule.content and rule.content.strip().lower() in text.lower():
            matches += 1
    elapsed = time.perf_counter() - start
    rule.last_tested = datetime.utcnow()
    rule.matches_count = matches
    await session.commit()
    return {"ruleId": id, "success": True, "matches": matches, "executionTime": round(elapsed, 3), "testedAt": datetime.utcnow().isoformat()}
