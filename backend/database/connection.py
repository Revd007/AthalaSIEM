from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings
from sqlalchemy.sql import text
import logging

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
    """Initialize database and run migrations if needed"""
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Run role updates
        update_roles_sql = """
        IF NOT EXISTS (SELECT * FROM dbo.users WHERE username = 'admin')
        BEGIN
            INSERT INTO dbo.users (
                id, 
                username, 
                email, 
                password_hash, 
                full_name, 
                role, 
                is_active
            )
            VALUES (
                NEWID(),
                'admin',
                'admin@athala.com',
                -- Hash for password 'admin123'
                '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGRH3GAzB.q',
                'Administrator',
                'ADMIN',
                1
            )
        END
        """
        try:
            await conn.execute(text(update_roles_sql))
            await conn.commit()
        except Exception as e:
            logging.error(f"Error updating roles: {e}")
            await conn.rollback()

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