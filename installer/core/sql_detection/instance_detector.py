from typing import Dict, Any, List, Optional
import winreg
import subprocess
import pyodbc
import logging
from pathlib import Path

class SQLInstanceDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def get_sql_instances(self) -> Dict[str, Any]:
        """Get detailed information about all SQL Server instances"""
        try:
            instances = []
            
            # Check registry for instances
            registry_instances = self._get_registry_instances()
            
            # Get detailed info for each instance
            for instance in registry_instances:
                instance_info = await self._get_detailed_instance_info(instance)
                if instance_info:
                    instances.append(instance_info)
            
            return {
                'status': 'success',
                'instances': instances,
                'count': len(instances),
                'has_express': any(i['edition'] == 'Express' for i in instances),
                'has_enterprise': any(i['edition'] == 'Enterprise' for i in instances)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting SQL instances: {e}")
            return {'status': 'error', 'error': str(e)}

    def _get_registry_instances(self) -> List[str]:
        """Get SQL Server instances from registry"""
        instances = []
        try:
            # Check 64-bit registry
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Microsoft SQL Server",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            ) as key:
                # Get installed instances
                instances.extend(winreg.QueryValueEx(key, "InstalledInstances")[0])
                
                # Get instance details
                for instance in instances:
                    instance_key = f"SOFTWARE\\Microsoft\\Microsoft SQL Server\\{instance}\\Setup"
                    try:
                        with winreg.OpenKey(
                            winreg.HKEY_LOCAL_MACHINE,
                            instance_key,
                            0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY
                        ) as ikey:
                            edition = winreg.QueryValueEx(ikey, "Edition")[0]
                            version = winreg.QueryValueEx(ikey, "Version")[0]
                    except WindowsError:
                        continue
                        
        except WindowsError as e:
            self.logger.warning(f"Registry access error: {e}")
            
        return instances

    async def _get_detailed_instance_info(self, instance: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a SQL Server instance"""
        try:
            # Try to connect to instance
            conn_str = f"Driver={{SQL Server}};Server={instance};Trusted_Connection=yes;"
            with pyodbc.connect(conn_str, timeout=3) as conn:
                cursor = conn.cursor()
                
                # Get version info
                cursor.execute("""
                    SELECT 
                        SERVERPROPERTY('ProductVersion') as Version,
                        SERVERPROPERTY('Edition') as Edition,
                        SERVERPROPERTY('ProductLevel') as ServicePack,
                        SERVERPROPERTY('InstanceName') as InstanceName,
                        @@VERSION as FullVersion
                """)
                row = cursor.fetchone()
                
                # Get instance status
                cursor.execute("SELECT state_desc FROM sys.databases WHERE name = 'master'")
                status = cursor.fetchone()[0]
                
                return {
                    'instance_name': instance,
                    'version': row.Version,
                    'edition': row.Edition,
                    'service_pack': row.ServicePack,
                    'status': status,
                    'full_version': row.FullVersion,
                    'available': True
                }
                
        except Exception as e:
            self.logger.warning(f"Could not get details for instance {instance}: {e}")
            return {
                'instance_name': instance,
                'available': False,
                'error': str(e)
            }