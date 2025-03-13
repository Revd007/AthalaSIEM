from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseCollector(ABC):
    """Base class for all collectors"""
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if collector is available"""
        pass
        
    @abstractmethod
    async def collect_events(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Collect events from source"""
        pass
        
    @abstractmethod
    async def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status and metrics"""
        pass