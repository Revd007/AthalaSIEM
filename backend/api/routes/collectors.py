from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from database.connection import get_db
from auth.dependencies.auth_handler import AuthHandler
from database.models import User
from ai_engine.donquixote_service import DonquixoteService

router = APIRouter(prefix="/api/collectors", tags=["collectors"])
ai_service = DonquixoteService()

@router.post("/start")
async def start_collectors(
    collector_types: Optional[List[str]] = None,
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Start specified collectors or all collectors if none specified"""
    try:
        result = await ai_service.start_collectors(collector_types)
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_collectors(
    collector_types: Optional[List[str]] = None,
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Stop specified collectors or all collectors"""
    try:
        result = await ai_service.stop_collectors(collector_types)
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_collectors_status(
    collector_type: Optional[str] = None,
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get status of specified collector or all collectors"""
    try:
        result = await ai_service.get_collector_status(collector_type)
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/{collector_type}")
async def get_collector_logs(
    collector_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    filters: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get logs from specified collector with optional time range and filters"""
    try:
        result = await ai_service.get_logs(
            collector_type=collector_type,
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_available_collectors(
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get list of available collector types"""
    try:
        collectors = {
            'windows': 'Windows Event Logs',
            'linux': 'Linux System Logs',
            'network': 'Network Device Logs',
            'cloud': 'Cloud Service Logs',
            'macos': 'MacOS System Logs',
            'evidence': 'Forensic Evidence'
        }
        return {
            'status': 'success',
            'collectors': collectors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/{collector_type}")
async def get_collector_config(
    collector_type: str,
    current_user: User = Depends(AuthHandler.get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get configuration options for specified collector"""
    try:
        if collector_type not in ai_service.collectors:
            raise HTTPException(status_code=404, detail=f"Collector {collector_type} not found")
            
        config = await ai_service.collectors[collector_type].get_config()
        return {
            'status': 'success',
            'collector': collector_type,
            'config': config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))