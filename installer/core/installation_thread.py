import asyncio
import json
from PyQt5.QtCore import QThread, pyqtSignal
import os
import shutil
import subprocess
import win32serviceutil
import win32service
import winreg
import logging
import pyodbc

from installer.core.db_validator import DatabaseValidator

class InstallationThread(QThread):
    progress_updated = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, install_path: str, port: int):
        super().__init__()
        self.install_path = install_path
        self.port = port
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        try:
            # Create installation directory
            self.progress_updated.emit(10)
            os.makedirs(self.install_path, exist_ok=True)
            
            # Copy application files
            self.progress_updated.emit(30)
            self._copy_application_files()
            
            # Configure database
            self.progress_updated.emit(50)
            self._setup_database()
            
            # Create configuration
            self.progress_updated.emit(70)
            self._create_config()
            
            # Install and start service
            self.progress_updated.emit(90)
            self._install_service()
            
            self.progress_updated.emit(100)
            
        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            self.error_occurred.emit(str(e))
            
    def _copy_application_files(self):
        """Copy application files to installation directory"""
        source_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        for item in os.listdir(source_dir):
            if item not in [".git", "__pycache__", "installer"]:
                src = os.path.join(source_dir, item)
                dst = os.path.join(self.install_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                    
    def _setup_database(self):
        """Setup database based on selected type"""
        # Create database validator
        validator = DatabaseValidator()
        
        # Validate SQL Server connection
        result = asyncio.run(validator.validate_connection(
            server="localhost\\SQLEXPRESS",
            user="sa", 
            password="AthalaSIEM2023!"
        ))
        
        if result['status'] != 'success':
            raise Exception(f"Database connection failed: {result['message']}")
            
        # Create database and tables
        conn_str = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=localhost\\SQLEXPRESS;"
            "UID=sa;PWD=AthalaSIEM2023!"
        )
        
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE AthalaSIEM")
            cursor.execute("USE AthalaSIEM")
            
            # Create tables (simplified example)
            cursor.execute("""
                CREATE TABLE events (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    timestamp DATETIME,
                    source VARCHAR(255),
                    event_type VARCHAR(50),
                    description TEXT
                )
            """)
            
            cursor.commit()
        pass
        
    def _create_config(self):
        """Create configuration file"""
        config = {
            "port": self.port,
            "install_path": self.install_path,
            "log_path": os.path.join(self.install_path, "logs"),
            "database": {
                "type": "MSSQL",
                "connection_string": "..."
            }
        }
        
        config_path = os.path.join(self.install_path, "config", "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            
    def _install_service(self):
        """Install and start Windows service"""
        service_path = os.path.join(self.install_path, "services", "windows_service.py")
        
        # Install service
        subprocess.run([
            "python", service_path, "install",
            "--startup", "auto"
        ], check=True)
        
        # Start service
        win32serviceutil.StartService("AthalaSIEM")