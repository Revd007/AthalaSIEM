from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from auth.dependencies.auth_handler import AuthHandler
from database.models import User
from ai_engine.donquixote_service import DonquixoteService
from utils.service_manager import ServiceManager
from app_config import settings

router = APIRouter(prefix="/api/service", tags=["Service Management"])
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)
service_manager = ServiceManager()

@router.get("/status")
async def get_service_status(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Get service status"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return await service_manager.get_status()

@router.post("/start")
async def start_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Start the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = await service_manager.start_service()
    return {"success": success}

@router.post("/stop")
async def stop_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Stop the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = await service_manager.stop_service()
    return {"success": success}

@router.post("/restart")
async def restart_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Restart the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = await service_manager.restart_service()
    return {"success": success}

@router.post("/ai/enable")
async def enable_ai(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Enable AI service"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can enable/disable AI service"
        )
    
    try:
        ai_service = DonquixoteService()
        result = await ai_service.ai_service_manager.enable_ai()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enable AI service: {str(e)}"
        )

@router.post("/ai/disable")
async def disable_ai(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Disable AI service"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can enable/disable AI service"
        )
    
    try:
        ai_service = DonquixoteService()
        result = await ai_service.ai_service_manager.disable_ai()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to disable AI service: {str(e)}"
        )

@router.get("/ai/status")
async def get_ai_status(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Get AI service status"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.ai_service_manager.get_ai_status()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI service status: {str(e)}"
        )