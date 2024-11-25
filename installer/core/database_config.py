from typing import Dict, Any, Optional
import psycopg2
from sqlalchemy import create_engine
import logging
from pathlib import Path

class DatabaseConfigManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def configure_database(self) -> Dict[str, Any]:
        """Configure PostgreSQL database"""
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
        conn = psycopg2.connect(
            host=self.config['host'],
            user=self.config['user'],
            password=self.config['password']
        )
        conn.autocommit = True
        
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {self.config['database_name']}")

    async def _create_app_user(self) -> None:
        """Create application database user"""
        conn = psycopg2.connect(
            host=self.config['host'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config['database_name']
        )
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Create user
            cur.execute(f"""
                CREATE USER {self.config['app_user']} WITH PASSWORD '{self.config['app_password']}';
                GRANT ALL PRIVILEGES ON DATABASE {self.config['database_name']} TO {self.config['app_user']};
                GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {self.config['app_user']};
                ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {self.config['app_user']};
            """)