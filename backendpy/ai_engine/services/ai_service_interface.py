from typing import Dict, Any, Protocol

class AIServiceInterface(Protocol):
    async def get_status(self) -> Dict[str, Any]:
        """Get AI service status"""
        ...

    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph data"""
        ...

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        ...

    async def get_recent_events(self) -> Dict[str, Any]:
        """Get recent security events"""
        ... 