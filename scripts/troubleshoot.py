import socket
from installer.core.error_handler import InstallationErrorHandler
from installer.core.sql_detector import SQLServerDetector
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

def check_database_connection():
    try:
        # Check SQL Server connection
        sql_detector = SQLServerDetector()
        sql_info = sql_detector.detect_sql_server()
        
        if not sql_info['installed']:
            return {"status": "error", "message": "SQL Server not installed"}
            
        # Verify at least one instance is working
        working_instances = []
        for instance in sql_info['instances']:
            if instance and 'version' in instance:
                working_instances.append(instance['instance_name'])
                
        if working_instances:
            return {
                "status": "ok", 
                "message": f"Connected to instances: {', '.join(working_instances)}"
            }
        
        return {"status": "error", "message": "No working SQL Server instances found"}
        # For example: attempt to connect to your database
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_required_ports():
    required_ports = [514, 1514, 1515, 55000]  # Example ports for SIEM
    open_ports = []
    blocked_ports = []
    
    for port in required_ports:
        try:
            # Basic port check using socket
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                open_ports.append(port)
            else:
                blocked_ports.append(port)
        except:
            blocked_ports.append(port)
    
    return {
        "status": "ok" if not blocked_ports else "error",
        "open_ports": open_ports,
        "blocked_ports": blocked_ports
    }

def fix_common_issues():
    error_handler = InstallationErrorHandler()
    # Fix common installation issues
    fixes = []
    
    # Check and fix services
    service_mgr = ServiceManager("AthalaSIEM")
    service_status = service_mgr.get_status()
    
    if service_status.get("status") != "running":
        try:
            service_mgr.start_service()
            fixes.append("Started AthalaSIEM service")
        except Exception as e:
            error_handler.handle_error("service_start_failed", str(e))
            
    # Check and fix database
    sql_detector = SQLServerDetector()
    sql_info = sql_detector.detect_sql_server()
    
    if not sql_info['installed']:
        error_handler.handle_error("sql_not_installed")
    else:
        for instance in sql_info['instances']:
            if not instance or 'version' not in instance:
                error_handler.handle_error("sql_instance_error", instance['instance_name'])
                
    # Check and fix ports
    required_ports = [514, 1514, 1515, 55000]
    for port in required_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:
                error_handler.handle_error("port_blocked", str(port))
        except Exception as e:
            error_handler.handle_error("port_check_failed", f"{port}: {str(e)}")
            
    return fixes