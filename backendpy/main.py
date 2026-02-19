"""
AthalaSIEM Python AI Backend.
Reads from shared PostgreSQL (log_entries), runs .pkl models, writes to ai_* tables.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_config import settings
from db.engine import init_db
from ml.loader import get_model_service
from routers.ai_analysis import router as ai_analysis_router
from routers.detection_rules import router as detection_rules_router
from routers.threat_hunting import router as threat_hunting_router
from routers.threat_intel import router as threat_intel_router
from routers.playbooks import router as playbooks_router
from routers.websocket import router as websocket_router
from workers.log_analyzer import run_log_analyzer_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_worker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task
    logger.info("Starting AthalaSIEM Python backend...")
    await init_db()
    get_model_service()
    _worker_task = asyncio.create_task(run_log_analyzer_worker())
    logger.info("DB, ML models, and log analyzer worker ready.")
    yield
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down.")


app = FastAPI(
    title="AthalaSIEM AI API",
    description="AI inference and threat hunting backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AthalaSIEM AI API", "docs": "/docs"}


app.include_router(ai_analysis_router)
app.include_router(detection_rules_router)
app.include_router(threat_hunting_router)
app.include_router(threat_intel_router)
app.include_router(playbooks_router)
app.include_router(websocket_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
