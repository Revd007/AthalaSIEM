from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings
import logging
from urllib.parse import quote_plus
from sqlalchemy import text

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
        pool_size=5,
        max_overflow=10
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
            # Create enum type if not exists
            await conn.execute(
                text("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
                            CREATE TYPE user_role AS ENUM ('admin', 'analyst', 'operator', 'viewer');
                        END IF;
                    END $$;
                """)
            )
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

async def get_db():
    """Dependency for getting async database session"""
    session = AsyncSessionLocal()
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