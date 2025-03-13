import win32serviceutil
import win32service
import subprocess
import logging
from typing import Dict, Any

class ServiceManager:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            status = win32serviceutil.QueryServiceStatus(self.service_name)
            return {
                'status': self._get_status_string(status[1]),
                'running': status[1] == win32service.SERVICE_RUNNING,
                'pid': status[2] if status[1] == win32service.SERVICE_RUNNING else None
            }
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {'status': 'unknown', 'running': False, 'pid': None}

    def start_service(self) -> bool:
        """Start the service"""
        try:
            win32serviceutil.StartService(self.service_name)
            return True
        except Exception as e:
            self.logger.error(f"Error starting service: {e}")
            return False

    def stop_service(self) -> bool:
        """Stop the service"""
        try:
            win32serviceutil.StopService(self.service_name)
            return True
        except Exception as e:
            self.logger.error(f"Error stopping service: {e}")
            return False

    def restart_service(self) -> bool:
        """Restart the service"""
        try:
            win32serviceutil.RestartService(self.service_name)
            return True
        except Exception as e:
            self.logger.error(f"Error restarting service: {e}")
            return False

    def _get_status_string(self, status_code: int) -> str:
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