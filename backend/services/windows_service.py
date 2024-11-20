import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import uvicorn
from api.server import app

class AthalaSIEMService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AthalaSIEM"
    _svc_display_name_ = "AthalaSIEM Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_alive = True

    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.is_alive = False

    def SvcDoRun(self):
        """Run the service"""
        try:
            # Start FastAPI server
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8080,
                log_level="info"
            )
        except Exception as e:
            servicemanager.LogErrorMsg(str(e))