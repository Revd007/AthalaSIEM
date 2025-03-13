from pathlib import Path
import yaml
from typing import Dict, Any
from cryptography.fernet import Fernet

class ConfigLoader:
    def __init__(self):
        self.config_dir = Path("config")
        self._load_config()
        
    def _load_config(self):
        """Load configuration files"""
        # Load main config
        with open(self.config_dir / "config.yaml") as f:
            self.config = yaml.safe_load(f)
            
        # Load secure config if exists
        secure_config_path = self.config_dir / "secure.conf"
        if secure_config_path.exists():
            with open(secure_config_path, "rb") as f:
                encrypted_data = f.read()
                self.secure_config = self._decrypt_config(encrypted_data)
                
    def get_database_url(self) -> str:
        """Get database connection URL"""
        db = self.config["database"]
        password = self.secure_config["database"]["password"]
        return f"postgresql://{db['user']}:{password}@{db['host']}:{db['port']}/{db['name']}" 