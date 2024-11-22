import requests
import os
import subprocess
import logging
from typing import Dict, Any, Optional
from packaging import version
import json
from pathlib import Path

class UpdateManager:
    def __init__(self, current_version: str, config: Dict[str, Any]):
        self.current_version = version.parse(current_version)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.update_url = config.get('update_url', 'https://api.example.com/updates')
        
    async def check_for_updates(self) -> Dict[str, Any]:
        """Check for available updates"""
        try:
            response = requests.get(f"{self.update_url}/latest")
            if response.status_code == 200:
                latest = response.json()
                latest_version = version.parse(latest['version'])
                
                return {
                    'current_version': str(self.current_version),
                    'latest_version': str(latest_version),
                    'update_available': latest_version > self.current_version,
                    'release_notes': latest.get('release_notes'),
                    'download_url': latest.get('download_url')
                }
        except Exception as e:
            self.logger.error(f"Failed to check for updates: {e}")
            return {
                'current_version': str(self.current_version),
                'error': str(e)
            }
            
    async def download_update(self, version: str) -> Optional[str]:
        """Download update package"""
        try:
            download_path = Path("updates") / f"update_{version}.zip"
            download_path.parent.mkdir(exist_ok=True)
            
            response = requests.get(
                f"{self.update_url}/download/{version}",
                stream=True
            )
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return str(download_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download update: {e}")
            return None