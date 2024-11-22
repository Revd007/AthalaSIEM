from typing import Dict, Any, Optional
import winreg
import subprocess
import logging
from pathlib import Path

class SQLServerDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_sql_server(self) -> Dict[str, Any]:
        """Detect existing SQL Server installations"""
        try:
            instances = self._get_sql_instances()
            if not instances:
                return {
                    'installed': False,
                    'instances': []
                }
                
            versions = []
            for instance in instances:
                version_info = self._get_instance_info(instance)
                if version_info:
                    versions.append(version_info)
                    
            return {
                'installed': True,
                'instances': versions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting SQL Server: {e}")
            return {
                'installed': False,
                'error': str(e)
            }
            
    def _get_sql_instances(self) -> List[str]:
        """Get list of installed SQL Server instances"""
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Microsoft SQL Server",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            ) as key:
                instances_path = winreg.QueryValueEx(key, "InstalledInstances")[0]
                return instances_path if isinstance(instances_path, list) else []
        except WindowsError:
            return []
            
    def _get_instance_info(self, instance: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about SQL Server instance"""
        try:
            cmd = f"sqlcmd -S {instance} -Q \"SELECT @@VERSION\""
            result = subprocess.check_output(cmd, shell=True).decode()
            
            return {
                'instance_name': instance,
                'version': self._parse_version(result),
                'edition': self._parse_edition(result)
            }
        except:
            return None