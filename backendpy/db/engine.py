"""
Async SQLAlchemy engine and session for PostgreSQL.
Uses asyncpg; same database as .NET backend.
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app_config import settings
from db.tables import (
    Base,
    ProcessedLogId,
    AIAnomaly,
    AIPrediction,
    AIHunt,
    AIHuntFinding,
    DetectionRule,
    ThreatIntelIndicator,
    PlaybookDefinition,
    PlaybookExecution,
)

# Only these tables are created by Python (log_entries is owned by .NET)
_OUR_TABLES = [
    ProcessedLogId.__table__,
    AIAnomaly.__table__,
    AIPrediction.__table__,
    AIHunt.__table__,
    AIHuntFinding.__table__,
    DetectionRule.__table__,
    ThreatIntelIndicator.__table__,
    PlaybookDefinition.__table__,
    PlaybookExecution.__table__,
]

# Ensure URL uses asyncpg driver
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    db_url,
    echo=False,
    poolclass=NullPool,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
        finally:
            await session.close()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _create_our_tables(sync_conn):
    for table in _OUR_TABLES:
        table.create(sync_conn, checkfirst=True)


async def init_db() -> None:
    """Create AI tables if not exist. Do not touch log_entries (owned by .NET)."""
    async with engine.begin() as conn:
        await conn.run_sync(_create_our_tables)
    # Seed threat intel indicators if empty (so dashboard shows non-zero)
    from sqlalchemy import select, func
    async with async_session_factory() as session:
        count = (await session.execute(select(func.count(ThreatIntelIndicator.id)))).scalar() or 0
        if count == 0:
            for val, typ in [
                ("192.168.1.1", "ip"),
                ("example-malware.com", "domain"),
                ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "hash"),
            ]:
                session.add(
                    ThreatIntelIndicator(
                        id=str(__import__("uuid").uuid4()),
                        type=typ,
                        value=val,
                        source_feed="seed",
                        confidence=0.5,
                    )
                )
            await session.commit()
