import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from .models.base import Base
import logging

class DatabaseManager:
    def __init__(self, database_url: str):
        # Convert URL format if needed
        if not database_url.startswith('mssql+aioodbc://'):
            if 'mssql' in database_url:
                database_url = database_url.replace('mssql://', 'mssql+aioodbc://')
            else:
                database_url = f"mssql+aioodbc://{database_url}"

        # Add driver parameter if not present
        if 'driver=' not in database_url.lower():
            if '?' in database_url:
                database_url += "&driver=ODBC+Driver+17+for+SQL+Server"
            else:
                database_url += "?driver=ODBC+Driver+17+for+SQL+Server"

        self.async_engine = create_async_engine(
            database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.async_session = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    @asynccontextmanager
    async def get_session(self):
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()

    async def init_models(self):
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

# Global engine and session factory
engine = None
AsyncSessionLocal = None

def init_db(database_url: str):
    global engine, AsyncSessionLocal
    
    if 'mssql' in database_url and '+aioodbc' not in database_url:
        database_url = database_url.replace('mssql://', 'mssql+aioodbc://')
    if 'driver=' not in database_url.lower():
        database_url += "?driver=ODBC+Driver+17+for+SQL+Server"
    
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True
    )
    
    AsyncSessionLocal = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

@asynccontextmanager
async def get_db():
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
        
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()