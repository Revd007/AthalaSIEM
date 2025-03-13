from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from core.services import ModelService
from database.models import ModelVersion
from ai_engine.donquixote_service import DonquixoteService

router = APIRouter(prefix="/model", tags=["Model Management"])

@router.post("/upgrade")
async def upgrade_model(
    config: Dict[str, Any],
    model_service: ModelService = Depends()
):
    """Upgrade model dengan konfigurasi baru"""
    try:
        result = await model_service.upgrade_model(config)
        return {
            "status": "success",
            "message": "Model upgraded successfully",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback/{version}")
async def rollback_model(
    version: str,
    model_service: ModelService = Depends()
):
    """Rollback model ke versi tertentu"""
    try:
        result = await model_service.rollback_model(version)
        return {
            "status": "success",
            "message": f"Model rolled back to version {version}",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions")
async def get_model_versions(
    model_service: ModelService = Depends()
):
    """Dapatkan semua versi model"""
    try:
        versions = await model_service.get_all_versions()
        return {
            "status": "success",
            "data": versions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_model_health(
    model_service: ModelService = Depends()
):
    """Dapatkan metrics kesehatan model"""
    try:
        metrics = await model_service.get_health_metrics()
        return {
            "status": "success",
            "model_name": "Donquixote Athala",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))