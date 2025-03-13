from contextlib import asynccontextmanager
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from auth.routes import auth
from api.routes import (
    alerts, events, users, playbooks, 
    system, dashboard, collectors, agents
)
from api.routes import ai_service as ai_service_router
from database.connection import init_db
from database.models.user import UserRole
from database.settings import settings as db_settings
from config import settings
from middleware.ssl_middleware import SSLMiddleware
from ai_engine.donquixote_service import DonquixoteService
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_service = DonquixoteService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="AthalaSIEM API",
    description="Security Information and Event Management API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add SSL middleware only if SSL is configured and enabled
if settings.ssl.SSL_ENABLED and os.path.exists(settings.ssl.SSL_CERT_PATH):
    app.add_middleware(
        SSLMiddleware,
        ssl_enabled=True,
        ssl_redirect=True,
        ssl_host=None
    )
    logger.info("SSL enabled with certificate")
else:
    logger.info("Running without SSL")

# Add routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(playbooks.router, prefix="/api/playbooks", tags=["Playbooks"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(collectors.router, prefix="/api/collectors", tags=["Collectors"])
app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])
app.include_router(ai_service_router.router, prefix="/api/ai", tags=["AI Service"])

@app.get("/")
async def root():
    return {"message": "AthalaSIEM API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/ai/status")
async def root_ai_status():
    return {"message": "Please use /api/ai/status instead"}

@app.post("/ai/analyze")
async def analyze_event(event_data: Dict[str, Any]):
    """Analyze event using AI service"""
    if not ai_service:
        return {
            "status": "error",
            "error": "AI service not initialized"
        }
    return await ai_service.analyze_event(event_data)