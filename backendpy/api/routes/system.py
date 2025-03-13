from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import psutil
import platform
from datetime import datetime

from database.connection import get_db
from ..schemas import SystemStatus, SystemMetrics

router = APIRouter()

@router.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status"""
    try:
        return {
            "status": "running",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "system_info": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.utcnow(),
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "usage_percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "usage_percent": disk.percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/processes")
async def get_system_processes():
    """Get list of running system processes"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            processes.append(proc.info)
        return {"processes": processes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/services")
async def get_system_services():
    """Get status of system services"""
    try:
        services = {
            "database": "running",
            "ai_engine": "running",
            "playbook_engine": "running",
            "alert_manager": "running"
        }
        return services
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))