"""
Playbooks: CRUD + run. Executions stored in playbook_executions.
"""
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_async_session
from db.tables import PlaybookDefinition, PlaybookExecution

router = APIRouter(prefix="/api/playbooks", tags=["Playbooks"])


class PlaybookCreate(BaseModel):
    name: str
    description: str = ""
    category: str = ""
    steps: list[dict] | None = None


class PlaybookUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    steps: list[dict] | None = None


@router.get("")
async def list_playbooks(session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(PlaybookDefinition).order_by(PlaybookDefinition.updated_at.desc()))
    items = r.scalars().all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "author": "system",
            "category": p.category,
            "status": "active",
            "steps": _parse_steps(p.steps_json),
            "lastModified": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in items
    ]


def _parse_steps(raw: str | None) -> list:
    if not raw:
        return []
    try:
        import json
        return json.loads(raw)
    except Exception:
        return []


@router.get("/{id}")
async def get_playbook(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(PlaybookDefinition).where(PlaybookDefinition.id == id))
    p = r.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Playbook not found")
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "author": "system",
        "category": p.category,
        "status": "active",
        "steps": _parse_steps(p.steps_json),
        "lastModified": p.updated_at.isoformat() if p.updated_at else None,
    }


@router.post("", status_code=201)
async def create_playbook(body: PlaybookCreate, session: AsyncSession = Depends(get_async_session)):
    import json
    steps_json = json.dumps(body.steps or [])
    p = PlaybookDefinition(
        id=str(uuid4()),
        name=body.name,
        description=body.description,
        category=body.category,
        steps_json=steps_json,
    )
    session.add(p)
    await session.commit()
    return {"id": p.id, "name": p.name, "description": p.description, "category": p.category, "steps": body.steps or []}


@router.put("/{id}")
async def update_playbook(id: str, body: PlaybookUpdate, session: AsyncSession = Depends(get_async_session)):
    import json
    r = await session.execute(select(PlaybookDefinition).where(PlaybookDefinition.id == id))
    p = r.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Playbook not found")
    if body.name is not None:
        p.name = body.name
    if body.description is not None:
        p.description = body.description
    if body.category is not None:
        p.category = body.category
    if body.steps is not None:
        p.steps_json = json.dumps(body.steps)
    p.updated_at = datetime.utcnow()
    await session.commit()
    return {"id": p.id, "name": p.name}


@router.delete("/{id}", status_code=204)
async def delete_playbook(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(PlaybookDefinition).where(PlaybookDefinition.id == id))
    p = r.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Playbook not found")
    await session.delete(p)
    await session.commit()


@router.post("/{id}/run")
async def run_playbook(id: str, session: AsyncSession = Depends(get_async_session)):
    r = await session.execute(select(PlaybookDefinition).where(PlaybookDefinition.id == id))
    p = r.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Playbook not found")
    steps = _parse_steps(p.steps_json)
    results = []
    for i, step in enumerate(steps):
        results.append({"stepIndex": i, "action": step.get("action", "query"), "status": "completed", "result": {}})
    ex = PlaybookExecution(
        id=str(uuid4()),
        playbook_id=p.id,
        status="completed",
        results_json=str(results),
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )
    session.add(ex)
    await session.commit()
    return {
        "id": ex.id,
        "playbookId": p.id,
        "playbookName": p.name,
        "status": ex.status,
        "startTime": ex.started_at.isoformat() if ex.started_at else None,
        "endTime": ex.completed_at.isoformat() if ex.completed_at else None,
        "results": results,
    }
