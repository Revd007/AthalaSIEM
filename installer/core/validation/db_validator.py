from typing import Dict, Any
import pyodbc
import logging
from pathlib import Path

class DatabaseRequirementsValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_version = "13.0"  # SQL Server 2016
        self.required_features = [
            'Database Engine Services',
            'Client Tools Connectivity'
        ]
        
    async def validate_instance(self, instance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SQL Server instance requirements"""
        try:
            validation_results = {
                'version_check': self._check_version(instance_info['version']),
                'edition_check': self._check_edition(instance_info['edition']),
                'features_check': await self._check_features(instance_info['instance_name']),
                'permissions_check': await self._check_permissions(instance_info),
                'performance_check': await self._check_performance(instance_info)
            }
            
            # Overall validation result
            validation_results['passed'] = all(
                check['passed'] for check in validation_results.values()
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {
                'passed': False,
                'error': str(e)
            }

    async def _check_permissions(self, instance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if required permissions are available"""
        required_permissions = [
            'CREATE DATABASE',
            'CREATE TABLE',
            'CREATE PROCEDURE',
            'BACKUP DATABASE',
            'EXECUTE'
        ]
        
        try:
            conn_str = f"Driver={{SQL Server}};Server={instance_info['instance_name']};Trusted_Connection=yes;"
            missing_permissions = []
            
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                for perm in required_permissions:
                    try:
                        cursor.execute(f"SELECT HAS_PERMS_BY_NAME(NULL, NULL, '{perm}')")
                        if not cursor.fetchone()[0]:
                            missing_permissions.append(perm)
                    except:
                        missing_permissions.append(perm)
                        
            return {
                'passed': len(missing_permissions) == 0,
                'missing_permissions': missing_permissions
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }