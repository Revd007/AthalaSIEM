import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from backend.database.models import Base
from backend.database.connection import engine
from backend.database.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_database():
    try:
        async with engine.begin() as conn:
            # Drop all existing tables (optional, uncomment if needed)
            # await conn.run_sync(Base.metadata.drop_all)
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create initial admin user
            create_admin_sql = """
            INSERT INTO public.users (
                id,
                username,
                email,
                password_hash,
                full_name,
                role,
                is_active
            ) VALUES (
                gen_random_uuid(),
                'admin',
                'admin@athala.com',
                '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiGRH3GAzB.q',
                'Administrator',
                'ADMIN',
                true
            ) ON CONFLICT (username) DO NOTHING;
            """
            await conn.execute(create_admin_sql)
            
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_database())
