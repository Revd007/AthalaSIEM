from typing import Dict, Any, Optional
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import logging
from pathlib import Path

class DatabaseConfigManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def configure_database(self) -> Dict[str, Any]:
        """Configure SQL Server database"""
        try:
            # Create database
            await self._create_database()
            
            # Configure permissions
            await self._configure_permissions()
            
            # Initialize schema
            await self._initialize_schema()
            
            # Create application user
            await self._create_app_user()
            
            return {
                'status': 'success',
                'database': self.config['database_name'],
                'user': self.config['app_user']
            }
            
        except Exception as e:
            self.logger.error(f"Database configuration failed: {e}")
            raise

    async def _create_database(self) -> None:
        """Create application database"""
        engine = create_engine(
            f"mssql+pyodbc://sa:{self.config['sa_password']}@"
            f"localhost\\SQLEXPRESS?driver=ODBC+Driver+17+for+SQL+Server"
        )
        
        with engine.connect() as conn:
            conn.execute(f"CREATE DATABASE {self.config['database_name']}")

    async def _create_app_user(self) -> None:
        """Create application database user"""
        engine = self._get_db_engine()
        
        with engine.connect() as conn:
            # Create login
            conn.execute(f"""
                CREATE LOGIN {self.config['app_user']}
                WITH PASSWORD = '{self.config['app_password']}'
            """)
            
            # Create user and assign roles
            conn.execute(f"""
                USE {self.config['database_name']};
                CREATE USER {self.config['app_user']}
                FOR LOGIN {self.config['app_user']};
                ALTER ROLE db_owner ADD MEMBER {self.config['app_user']};
            """)