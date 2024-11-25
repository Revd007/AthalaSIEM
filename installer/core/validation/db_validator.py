from typing import Dict, Any
import psycopg2
import logging
from pathlib import Path

class DatabaseRequirementsValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_version = "12.0"  # PostgreSQL 12
        
    async def validate_instance(self, instance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PostgreSQL instance requirements"""
        try:
            validation_results = {
                'version_check': self._check_version(instance_info),
                'permissions_check': await self._check_permissions(instance_info),
                'connection_check': await self._check_connection(instance_info)
            }
            
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
            'CREATE ROLE'
        ]
        
        try:
            conn = psycopg2.connect(
                host=instance_info['host'],
                user=instance_info['user'],
                password=instance_info['password']
            )
            
            with conn.cursor() as cur:
                cur.execute("SELECT current_user;")
                current_user = cur.fetchone()[0]
                
                cur.execute(f"""
                    SELECT privilege_type 
                    FROM information_schema.role_usage_grants 
                    WHERE grantee = '{current_user}'
                """)
                granted_permissions = [r[0] for r in cur.fetchall()]
                
                missing_permissions = [
                    p for p in required_permissions 
                    if p not in granted_permissions
                ]
                
            return {
                'passed': len(missing_permissions) == 0,
                'missing_permissions': missing_permissions
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }