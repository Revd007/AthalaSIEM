from installer.core.error_handler import InstallationErrorHandler
from utils.service_manager import ServiceManager
import logging

def check_installation():
    # Check services
    service_mgr = ServiceManager("AthalaSIEM")
    service_status = service_mgr.get_status()
    
    # Check database
    db_status = check_database_connection()
    
    # Check ports
    ports_status = check_required_ports()
    
    return {
        "service": service_status,
        "database": db_status,
        "ports": ports_status
    }

def fix_common_issues():
    error_handler = InstallationErrorHandler()
    # Implementasi perbaikan masalah umum