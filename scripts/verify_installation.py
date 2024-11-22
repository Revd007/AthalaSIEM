from api.middleware import logging
from installer.core.sql_detector import SQLServerDetector


async def check_services():
    from utils.service_manager import ServiceManager
    
    service_mgr = ServiceManager("AthalaSIEM")
    service_status = service_mgr.get_status()
    
    if service_status.get("status") != "running":
        return False
    return True

async def check_database():
    try:
        # Check SQL Server connection
        sql_detector = SQLServerDetector()
        sql_info = sql_detector.detect_sql_server()
        
        if not sql_info['installed']:
            return False
            
        # Verify at least one instance is working
        for instance in sql_info['instances']:
            if instance and 'version' in instance:
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"Database verification failed: {e}")
        return False
    return True

async def check_collectors():
    try:
        # Check if collector services are running
        from utils.service_manager import ServiceManager
        
        collector_services = [
            "AthalaSIEM_Syslog",
            "AthalaSIEM_WinEventLog", 
            "AthalaSIEM_FileMonitor"
        ]
        
        service_mgr = ServiceManager()
        
        for service in collector_services:
            status = service_mgr.get_status(service)
            if status.get("status") != "running":
                logging.error(f"Collector service {service} is not running")
                return False
                
        # Check collector ports
        import socket
        collector_ports = [514, 1514, 1515]  # Syslog, WinEvent, FileMonitor ports
        
        for port in collector_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:
                logging.error(f"Collector port {port} is not open")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Collector verification failed: {e}")
        return False
    return True

async def check_ai_engine():
    try:
        # Check if AI engine service is running
        from utils.service_manager import ServiceManager
        
        service_mgr = ServiceManager()
        ai_service_status = service_mgr.get_status("AthalaSIEM_AIEngine")
        
        if ai_service_status.get("status") != "running":
            logging.error("AI Engine service is not running")
            return False
            
        # Check AI model files exist
        import os
        model_path = os.path.join(os.environ.get("ATHALA_HOME", ""), "ai", "models")
        required_models = ["threat_detection.pkl", "anomaly_detection.pkl"]
        
        for model in required_models:
            if not os.path.exists(os.path.join(model_path, model)):
                logging.error(f"AI model file {model} not found")
                return False
                
        # Check AI engine port
        import socket
        ai_port = 55000
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        result = sock.connect_ex(('127.0.0.1', ai_port))
        sock.close()
        
        if result != 0:
            logging.error(f"AI Engine port {ai_port} is not open")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"AI Engine verification failed: {e}")
        return False
    return True

async def check_web_interface():
    try:
        # Check if web service is running
        from utils.service_manager import ServiceManager
        
        service_mgr = ServiceManager()
        web_status = service_mgr.get_status("AthalaSIEM_WebUI")
        
        if web_status.get("status") != "running":
            logging.error("Web interface service is not running")
            return False
            
        # Check web server port
        import socket
        web_port = 443 # HTTPS port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', web_port))
        sock.close()
        
        if result != 0:
            logging.error(f"Web interface port {web_port} is not open")
            return False
            
        # Check static files exist
        import os
        web_path = os.path.join(os.environ.get("ATHALA_HOME", ""), "web")
        required_files = ["index.html", "styles/main.css", "js/app.js"]
        
        for file in required_files:
            if not os.path.exists(os.path.join(web_path, file)):
                logging.error(f"Web interface file {file} not found")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Web interface verification failed: {e}")
        return False
    return True

async def verify_installation():
    checks = [
        await check_services(),
        await check_database(),
        await check_collectors(),
        await check_ai_engine(),
        await check_web_interface()
    ]
    
    return all(checks)