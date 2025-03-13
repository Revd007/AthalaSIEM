import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import uvicorn
from fastapi import FastAPI
from database.connection import init_db
from database.settings import settings
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

class AthalaSIEMService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AthalaSIEM"
    _svc_display_name_ = "AthalaSIEM Service"
    _svc_description_ = "AthalaSIEM Security Information and Event Management Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.is_alive = True

    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.is_alive = False

    def SvcDoRun(self):
        """Run the service"""
        try:
            # Initialize database
            init_db()
            
            # Start FastAPI server
            app = FastAPI(
                title=settings.PROJECT_NAME,
                version=settings.PROJECT_VERSION
            )
            
            uvicorn.run(
                app,
                host="0.0.0.0", 
                port=settings.PORT,
                ssl_keyfile=settings.SSL_KEYFILE if settings.USE_HTTPS else None,
                ssl_certfile=settings.SSL_CERTFILE if settings.USE_HTTPS else None
            )
        except Exception as e:
            servicemanager.LogErrorMsg(str(e))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AthalaSIEMService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(AthalaSIEMService)