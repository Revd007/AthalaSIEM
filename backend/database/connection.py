from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import urllib.parse
import logging
from typing import Generator
from contextlib import contextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
DB_USER = os.getenv("DB_USER", "revian_dbsiem")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", "wokolcoy20"))
DB_HOST = os.getenv("DB_HOST", ".\SQLEXPRESS")
DB_NAME = os.getenv("DB_NAME", "siem_db")
DB_PORT = os.getenv("DB_PORT", "1433")

# Create connection string
SQLALCHEMY_DATABASE_URL = f"mssql+pyodbc://{DB_HOST}/{DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

# Create engine with connection pooling
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=10,        # Maximum number of connections in the pool
    max_overflow=20,     # Maximum number of connections that can be created beyond pool_size
    pool_timeout=30,     # Seconds to wait before giving up on getting a connection
    pool_recycle=1800,   # Recycle connections after 30 minutes
    echo=False           # Set to True to log all SQL queries (development only)
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

@contextmanager
def get_db() -> Generator:
    """Database session context manager"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()