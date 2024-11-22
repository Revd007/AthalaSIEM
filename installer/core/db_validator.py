class DatabaseValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def validate_connection(self, 
                                server: str,
                                user: str,
                                password: str) -> Dict[str, Any]:
        """Validate database connection"""
        try:
            # Test connection
            conn_str = (
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server={server};UID={user};PWD={password}"
            )
            
            with pyodbc.connect(conn_str, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0]
                
            return {
                'status': 'success',
                'message': 'Connection successful',
                'version': version
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def validate_requirements(self, instance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SQL Server meets requirements"""
        try:
            version = instance_info['version']
            edition = instance_info['edition']
            
            # Check version compatibility
            version_ok = self._check_version_compatibility(version)
            
            # Check edition features
            features_ok = self._check_edition_features(edition)
            
            return {
                'compatible': version_ok and features_ok,
                'version_ok': version_ok,
                'features_ok': features_ok,
                'message': self._get_validation_message(version_ok, features_ok)
            }
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {
                'compatible': False,
                'message': f"Error validating SQL Server: {e}"
            }