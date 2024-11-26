from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings
import logging
from urllib.parse import quote_plus

# Setup logging
logger = logging.getLogger(__name__)

Base = declarative_base()

# Escape special characters in password
db_password = quote_plus(settings.DB_PASSWORD)

# PostgreSQL async URL format with escaped password
async_database_url = f"postgresql+asyncpg://{settings.DB_USER}:{db_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

# Create async engine with better error handling
try:
    engine = create_async_engine(
        async_database_url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "server_settings": {
                "client_encoding": "utf8"
            }
        }
    )
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Initialize database and create tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

async def get_db():
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_models():
    """Initialize database models"""
    async with engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            await conn.rollback()