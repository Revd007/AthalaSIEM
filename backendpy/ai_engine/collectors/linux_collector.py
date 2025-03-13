from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
from .base_collector import BaseCollector

class LinuxCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.log_files = {
            'syslog': '/var/log/syslog',
            'auth': '/var/log/auth.log',
            'kern': '/var/log/kern.log',
            'daemon': '/var/log/daemon.log',
            'messages': '/var/log/messages'
        }

    async def check_availability(self) -> bool:
        """Check if Linux logs are accessible"""
        try:
            return any(Path(path).exists() and Path(path).is_file() 
                      for path in self.log_files.values())
        except Exception as e:
            self.logger.error(f"Error checking Linux logs availability: {e}")
            return False

    async def collect_events(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Collect Linux system logs"""
        # Gunakan implementasi yang sudah ada
        return await super().collect_events(hours)

    async def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status and metrics"""
        try:
            is_available = await self.check_availability()
            recent_events = await self.collect_events(hours=1)
            
            return {
                'status': 'active' if is_available else 'inactive',
                'collector_type': 'linux',
                'metrics': {
                    'events_collected': len(recent_events),
                    'log_files_available': sum(1 for path in self.log_files.values() 
                                             if Path(path).exists()),
                    'total_log_files': len(self.log_files),
                    'last_collection': datetime.utcnow().isoformat()
                },
                'errors': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collector status: {e}")
            return {
                'status': 'error',
                'collector_type': 'linux',
                'errors': [str(e)],
                'timestamp': datetime.utcnow().isoformat()
            }
    