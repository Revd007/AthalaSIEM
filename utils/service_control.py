import win32serviceutil
import win32service
import win32event
import win32api
import logging
from typing import Dict, Any

class ServiceController:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        
    def install(self, service_path: str) -> bool:
        """Install the service"""
        try:
            win32serviceutil.InstallService(
                pythonClassString=service_path,
                serviceName=self.service_name,
                startType=win32service.SERVICE_AUTO_START
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to install service: {e}")
            return False
            
    def uninstall(self) -> bool:
        """Uninstall the service"""
        try:
            win32serviceutil.RemoveService(self.service_name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to uninstall service: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            status = win32serviceutil.QueryServiceStatus(self.service_name)
            return {
                'running': status[1] == win32service.SERVICE_RUNNING,
                'status': self._get_status_string(status[1]),
                'pid': status[2] if status[1] == win32service.SERVICE_RUNNING else None
            }
        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {
                'running': False,
                'status': 'unknown',
                'pid': None
            }
            
    def _get_status_string(self, status_code: int) -> str:
        """Convert status code to string"""
        status_map = {
            win32service.SERVICE_STOPPED: 'stopped',
            win32service.SERVICE_START_PENDING: 'starting',
            win32service.SERVICE_STOP_PENDING: 'stopping',
            win32service.SERVICE_RUNNING: 'running',
            win32service.SERVICE_CONTINUE_PENDING: 'continuing',
            win32service.SERVICE_PAUSE_PENDING: 'pausing',
            win32service.SERVICE_PAUSED: 'paused'
        }
        return status_map.get(status_code, 'unknown')