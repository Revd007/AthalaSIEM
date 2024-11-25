import sqlalchemy
from sqlalchemy import create_engine
import subprocess
import os
from typing import Optional
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from backend.database.settings import settings
import logging

class DatabaseInitializer:
    def __init__(self, db_type: str, install_path: str):
        self.db_type = db_type
        self.install_path = install_path
        
    def initialize(self) -> Optional[str]:
        """Initialize database and return connection string"""
        if self.db_type == "SQLite":
            return self._init_sqlite()
        else:
            return self._init_postgresql()
            
    def _init_sqlite(self) -> str:
        """Initialize SQLite database"""
        db_path = os.path.join(self.install_path, "data", "siem.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        engine = create_engine(f"sqlite:///{db_path}")
        
        # Import and create all models
        from database.models import Base
        Base.metadata.create_all(engine)
        
        return f"sqlite:///{db_path}"
        
    def _init_postgresql(self) -> Optional[str]:
        """Initialize PostgreSQL database"""
        try:
            # Connect to PostgreSQL server
            conn = psycopg2.connect(
                host=settings.DB_HOST,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Create database if not exists
            cur = conn.cursor()
            cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{settings.DB_NAME}'")
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {settings.DB_NAME}")
            
            cur.close()
            conn.close()
            
            # Return connection string
            return f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQL: {e}")
            return None