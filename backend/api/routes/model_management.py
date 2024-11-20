from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from core.services import ModelService
from database.models import ModelVersion

router = APIRouter()

@router.post("/model/upgrade")
async def upgrade_model(
    config: Dict[str, Any],
    model_service: ModelService = Depends()
):
    """Endpoint untuk upgrade model"""
    try:
        result = await model_service.upgrade_model(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/rollback/{version}")
async def rollback_model(
    version: str,
    model_service: ModelService = Depends()
):
    """Endpoint untuk rollback model"""
    try:
        result = await model_service.rollback_model(version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/versions")
async def get_model_versions(
    model_service: ModelService = Depends()
):
    """Get all model versions"""
    versions = await model_service.get_all_versions()
    return versions