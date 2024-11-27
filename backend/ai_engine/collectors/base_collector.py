from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseCollector(ABC):
    @abstractmethod
    async def start_collection(self):
        """Start collecting logs"""
        pass

    @abstractmethod
    async def stop_collection(self):
        """Stop collecting logs"""
        pass

    @abstractmethod
    async def get_logs(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get collected logs within time range and filters"""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        pass