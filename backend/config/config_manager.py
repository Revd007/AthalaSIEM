from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

@dataclass
class ServiceConfig:
    name: str = "AthalaSIEM"
    display_name: str = "AthalaSIEM Service"
    description: str = "Security Information and Event Management Service"
    executable: str = "python.exe"
    script_path: str = "services/windows_service.py"
    startup_type: str = "auto"

@dataclass
class DatabaseConfig:
    type: str = "MSSQL"
    host: str = "localhost"
    name: str = "siem_db"
    user: str = "revian_dbsiem"
    password: str = "Wokolcoy@20"
    port: int = 1433
    auto_install: bool = True

class ConfigManager:
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_config()
                self.save_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = self._create_default_config()
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            "service": asdict(ServiceConfig()),
            "database": asdict(DatabaseConfig()),
            "web": {
                "port": 8080,
                "host": "0.0.0.0",
                "use_https": False,
                "ssl_cert": "certs/cert.pem",
                "ssl_key": "certs/key.pem"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/siem.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            }
        }