from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from auth.dependencies.auth_handler import AuthHandler
from database.models.user import User
from ai_engine.types import AIServiceInterface
from ai_engine.donquixote_service import DonquixoteService
from config import settings
from datetime import datetime

router = APIRouter(prefix="/api/ai", tags=["AI Service"])
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)

@router.get("/status")
async def get_ai_status():
    """Get AI service status"""
    try:
        ai_service = DonquixoteService()
        status = await ai_service.get_status()
        
        # Tambahkan data dummy untuk testing
        return {
            "data": {
                "status": {
                    "service_status": "active",
                    "model_performance": {
                        "accuracy": 0.95
                    },
                    "system_health": {
                        "score": 85
                    },
                    "events_analysis": {
                        "statistics": {
                            "high_risk_events": 12,
                            "active_alerts": 5,
                            "network_throughput": 95,
                            "security_score": 88,
                            "total_incidents": 45
                        },
                        "recent_events": [
                            {
                                "id": "evt1",
                                "timestamp": "2024-01-20T10:30:00Z",
                                "type": "Intrusion Attempt",
                                "severity": "high",
                                "source": "Firewall",
                                "message": "Multiple failed login attempts detected"
                            },
                            # Tambahkan beberapa event lagi
                        ],
                        "threat_patterns": [
                            {"name": "Malware", "value": 35},
                            {"name": "Phishing", "value": 25},
                            {"name": "DDoS", "value": 20},
                            {"name": "Data Breach", "value": 15}
                        ]
                    }
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI status: {str(e)}"
        )

@router.get("/knowledge-graph")
async def get_knowledge_graph():
    """Get knowledge graph data"""
    try:
        ai_service = DonquixoteService()
        graph_data = await ai_service.get_knowledge_graph()
        return {
            "data": {
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
                "patterns": graph_data.get("patterns", [])
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get knowledge graph: {str(e)}"
        )

@router.get("/system-metrics")
async def get_system_metrics():
    """Get system metrics"""
    try:
        ai_service = DonquixoteService()
        metrics = await ai_service.get_system_metrics()
        return {
            "data": {
                "cpu_usage": metrics.get("cpu_usage", 0),
                "memory_usage": metrics.get("memory_usage", 0),
                "disk_usage": metrics.get("disk_usage", 0),
                "network_throughput": metrics.get("network_throughput", 0),
                "last_updated": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )

@router.get("/recent-events")
async def get_recent_events():
    """Get recent security events"""
    try:
        ai_service = DonquixoteService()
        events = await ai_service.get_recent_events()
        return {
            "data": [
                {
                    "id": str(event.get("id", i)),
                    "type": event.get("type", "unknown"),
                    "severity": event.get("severity", "medium"),
                    "source": event.get("source", "system"),
                    "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
                    "description": event.get("description", "No description"),
                    "status": event.get("status", "new")
                }
                for i, event in enumerate(events)
            ]
        }
    except Exception as e:
        print(f"Error getting recent events: {e}")  # Debug log
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent events: {str(e)}"
        )

@router.post("/analyze/event")
async def analyze_event(
    event_data: Dict[str, Any],
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Analyze security event using AI"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.analyze_event(event_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze event: {str(e)}"
        )

@router.post("/analyze/threat")
async def analyze_threat(
    event_data: Dict[str, Any],
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Analyze threat in event"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.threat_intelligence.analyze_threats(event_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze threat: {str(e)}"
        )

@router.post("/analyze/anomaly")
async def analyze_anomaly(
    event_data: Dict[str, Any],
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Detect anomalies in event"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.anomaly_detection.detect_anomalies(event_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze anomaly: {str(e)}"
        )

@router.post("/analyze/behavior")
async def analyze_behavior(
    event_data: Dict[str, Any],
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Analyze behavior patterns in event"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.behavior_analyzer(event_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze behavior: {str(e)}"
        )

@router.post("/assess/risk")
async def assess_risk(
    event_data: Dict[str, Any],
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Assess risk level of event"""
    try:
        ai_service = DonquixoteService()
        return await ai_service.risk_assessor(event_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assess risk: {str(e)}"
        )

@router.get("/metrics")
async def get_ai_metrics(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Get AI performance metrics"""
    try:
        ai_service = DonquixoteService()
        return ai_service.metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.post("/train")
async def trigger_training(
    current_user: User = Depends(auth_handler.get_current_user)
) -> Dict[str, Any]:
    """Trigger AI model training"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can trigger training"
        )
    
    try:
        ai_service = DonquixoteService()
        await ai_service.ensure_model_ready()
        return {"status": "success", "message": "Training completed"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train model: {str(e)}"
        ) 