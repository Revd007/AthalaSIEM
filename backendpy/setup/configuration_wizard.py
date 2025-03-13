from typing import Dict, Any
import yaml
import os
from pathlib import Path
import secrets
from cryptography.fernet import Fernet

class SetupWizard:
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
    async def run_initial_setup(self) -> Dict[str, Any]:
        """Run initial setup wizard"""
        config = {
            "database": await self._setup_database(),
            "security": await self._setup_security(),
            "email": await self._setup_email(),
            "ssl": await self._setup_ssl(),
            "logging": await self._setup_logging()
        }
        
        # Save configuration
        await self._save_config(config)
        return config
        
    async def _setup_database(self) -> Dict[str, Any]:
        """Database configuration wizard"""
        print("\n=== Database Configuration ===")
        return {
            "host": input("Database Host [localhost]: ") or "localhost",
            "port": input("Database Port [5432]: ") or "5432",
            "name": input("Database Name [siem_db]: ") or "siem_db",
            "user": input("Database User: "),
            "password": input("Database Password: ")
        }
        
    async def _setup_security(self) -> Dict[str, Any]:
        """Security configuration wizard"""
        print("\n=== Security Configuration ===")
        
        # Generate secure defaults
        secret_key = secrets.token_hex(32)
        encryption_key = Fernet.generate_key().decode()
        
        return {
            "secret_key": secret_key,
            "encryption_key": encryption_key,
            "jwt_expiry_minutes": 30,
            "enable_ssl": input("Enable SSL? (y/n) [y]: ").lower() != 'n',
            "cors_origins": ["*"]  # Can be configured later
        }
        
    async def _setup_ssl(self) -> Dict[str, Any]:
        """SSL configuration wizard"""
        print("\n=== SSL Configuration ===")
        
        use_ssl = input("Use SSL? (y/n) [y]: ").lower() != 'n'
        if not use_ssl:
            return {"enabled": False}
            
        cert_type = input("SSL Certificate Type (self-signed/custom) [self-signed]: ") or "self-signed"
        
        if cert_type == "self-signed":
            # Generate self-signed certificate
            from cryptography import x509
            # ... certificate generation code ...
            return {
                "enabled": True,
                "type": "self-signed",
                "cert_path": "certs/cert.pem",
                "key_path": "certs/key.pem"
            }
        else:
            return {
                "enabled": True,
                "type": "custom",
                "cert_path": input("Certificate Path: "),
                "key_path": input("Private Key Path: ")
            }
            
    async def _setup_email(self) -> Dict[str, Any]:
        """Email configuration wizard"""
        print("\n=== Email Configuration (Optional) ===")
        
        use_email = input("Configure Email Notifications? (y/n) [n]: ").lower() == 'y'
        if not use_email:
            return {"enabled": False}
            
        return {
            "enabled": True,
            "server": input("SMTP Server: "),
            "port": int(input("SMTP Port [587]: ") or "587"),
            "username": input("SMTP Username: "),
            "password": input("SMTP Password: "),
            "from_name": input("From Name [AthalaSIEM]: ") or "AthalaSIEM",
            "from_email": input("From Email: "),
            "use_tls": True
        }
        
    async def _setup_logging(self) -> Dict[str, Any]:
        """Logging configuration wizard"""
        print("\n=== Logging Configuration ===")
        
        log_dir = input("Log Directory [logs]: ") or "logs"
        Path(log_dir).mkdir(exist_ok=True)
        
        return {
            "level": input("Log Level (DEBUG/INFO/WARNING/ERROR) [INFO]: ") or "INFO",
            "directory": log_dir,
            "max_size_mb": 10,
            "backup_count": 5
        }
        
    async def _save_config(self, config: Dict[str, Any]):
        """Save configuration securely"""
        # Save sensitive config securely
        sensitive_config = {
            "database": {
                "password": config["database"]["password"]
            },
            "security": {
                "secret_key": config["security"]["secret_key"],
                "encryption_key": config["security"]["encryption_key"]
            },
            "email": config["email"] if config["email"]["enabled"] else {}
        }
        
        # Encrypt sensitive config
        encrypted_config = self._encrypt_config(sensitive_config)
        with open(self.config_dir / "secure.conf", "wb") as f:
            f.write(encrypted_config)
            
        # Save non-sensitive config
        safe_config = {
            "database": {
                "host": config["database"]["host"],
                "port": config["database"]["port"],
                "name": config["database"]["name"],
                "user": config["database"]["user"]
            },
            "security": {
                "jwt_expiry_minutes": config["security"]["jwt_expiry_minutes"],
                "enable_ssl": config["security"]["enable_ssl"],
                "cors_origins": config["security"]["cors_origins"]
            },
            "ssl": config["ssl"],
            "logging": config["logging"]
        }
        
        with open(self.config_dir / "config.yaml", "w") as f:
            yaml.dump(safe_config, f) 