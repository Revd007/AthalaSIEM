from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from uuid import UUID

from ..schemas import PlaybookTemplateCreate, PlaybookTemplateResponse, PlaybookRunResponse
from database.connection import get_db
from database.models import PlaybookTemplate, PlaybookRun
from automation.playbook_engine import PlaybookEngine

router = APIRouter()
playbook_engine = PlaybookEngine()

@router.post("/playbooks/templates/", response_model=PlaybookTemplateResponse)
async def create_playbook_template(
    template: PlaybookTemplateCreate, 
    db: Session = Depends(get_db)
):
    db_template = PlaybookTemplate(**template.dict())
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template

@router.get("/playbooks/templates/", response_model=List[PlaybookTemplateResponse])
async def get_playbook_templates(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    templates = db.query(PlaybookTemplate).offset(skip).limit(limit).all()
    return templates

@router.get("/playbooks/templates/{template_id}", response_model=PlaybookTemplateResponse)
async def get_playbook_template(
    template_id: UUID, 
    db: Session = Depends(get_db)
):
    template = db.query(PlaybookTemplate).filter(PlaybookTemplate.id == template_id).first()
    if template is None:
        raise HTTPException(status_code=404, detail="Playbook template not found")
    return template

@router.post("/playbooks/run/{playbook_id}")
async def run_playbook(
    playbook_id: str,
    context: Dict[str, Any],
    db: Session = Depends(get_db)
):
    try:
        result = await playbook_engine.execute_playbook(playbook_id, context)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/playbooks/status/{run_id}", response_model=PlaybookRunResponse)
async def get_playbook_status(
    run_id: UUID,
    db: Session = Depends(get_db)
):
    status = await playbook_engine.get_playbook_status(str(run_id))
    if status['status'] == 'not_found':
        raise HTTPException(status_code=404, detail="Playbook run not found")
    return status

@router.post("/playbooks/stop/{run_id}")
async def stop_playbook(
    run_id: UUID,
    db: Session = Depends(get_db)
):
    result = await playbook_engine.stop_playbook(str(run_id))
    if result['status'] == 'not_found':
        raise HTTPException(status_code=404, detail="Playbook run not found")
    return result