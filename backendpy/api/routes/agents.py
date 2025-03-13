from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pytest import Session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid
from database.connection import get_db
from database.models.agent import Agent, AgentType, CollectorType
from schemas.agent import AgentCreate, AgentResponse
from services.agent_service import AgentService
from schemas.agent import AgentCreate, AgentUpdate, AgentResponse
from utils.installer_generator import generate_agent_installer

router = APIRouter(prefix="/api/agents", tags=["Agents"], redirect_slashes=False)
agent_service = AgentService()

@router.get("/", response_model=dict)
async def get_agents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get all agents"""
    agents = await agent_service.get_agents(db, skip, limit)
    return {
        "status": "success",
        "data": agents,
        "message": "Agents retrieved successfully"
    }

@router.post("/deploy", response_model=AgentResponse)
async def deploy_agent(agent_data: AgentCreate, db: AsyncSession = Depends(get_db)):
    """Deploy a new agent"""
    try:
        agent = await agent_service.create_agent(db, agent_data)
        return agent
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get agent by ID"""
    agent = await agent_service.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    agent_data: AgentUpdate,
    db: Session = Depends(get_db)
):
    """Update agent"""
    agent = await agent_service.update_agent(db, agent_id, agent_data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete agent"""
    success = await agent_service.delete_agent(db, agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "success", "message": "Agent deleted"}

@router.get("/installer/{agent_type}")
async def download_agent_installer(agent_type: str):
    """Download agent installer"""
    try:
        installer_path = generate_agent_installer(agent_type)
        return FileResponse(
            installer_path,
            media_type='application/octet-stream',
            filename=f'siem-agent-{agent_type}.zip'
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 