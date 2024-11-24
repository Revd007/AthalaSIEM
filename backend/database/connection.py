from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

Base = declarative_base()

# Convert MSSQL URL to async format
# Example: mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server
# becomes: mssql+aioodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server
async_database_url = settings.DATABASE_URL.replace('mssql+pyodbc', 'mssql+aioodbc')

# Create async engine
engine = create_async_engine(
    async_database_url,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_models():
    """Initialize database models"""
    from .models import Base  # Import your Base class
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)