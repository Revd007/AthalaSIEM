import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from .models.base import Base

# Use async SQL Server driver
SQLALCHEMY_DATABASE_URL = "mssql+aioodbc://revian_dbsiem:Wokolcoy@20@server:1433/siem_db?driver=ODBC+Driver+17+for+SQL+Server"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_recycle=3600
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@asynccontextmanager
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e

# Create all tables in the database
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Export Base and async session
__all__ = ['Base', 'engine', 'AsyncSessionLocal', 'get_db', 'init_db']