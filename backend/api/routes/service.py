from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from utils.service_manager import ServiceManager
from auth.dependencies.auth_handler import AuthHandler
from database.models import User, UserRole

router = APIRouter()
service_manager = ServiceManager("AthalaSIEM")
auth_handler = AuthHandler()

@router.get("/status")
async def get_service_status(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Get service status"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return service_manager.get_status()

@router.post("/start")
async def start_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Start the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = service_manager.start_service()
    return {"success": success}

@router.post("/stop")
async def stop_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Stop the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = service_manager.stop_service()
    return {"success": success}

@router.post("/restart")
async def restart_service(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict:
    """Restart the service"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    success = service_manager.restart_service()
    return {"success": success}