from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth.routes import auth
from api.routes import alerts, events, users, playbooks, system, dashboard, collectors
from database.connection import init_db
from database.models.user import UserRole
from database.settings import settings
import logging
from pathlib import Path
import yaml
from ai_engine.donquixote_service import DonquixoteService
from typing import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ai_logger = logging.getLogger('ai_engine')

# Initialize AI service globally
ai_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ai_service
    
    # Initialize AI components
    try:
        ai_config_path = Path("backend/ai_engine/services/ai_settings.yaml")
        if not ai_config_path.exists():
            ai_logger.warning("AI config file not found, using default settings")
            ai_service = DonquixoteService()
        else:
            with open(ai_config_path) as f:
                ai_config = yaml.safe_load(f)
                ai_service = DonquixoteService(config=ai_config['model'])
    except Exception as e:
        ai_logger.error(f"Failed to initialize AI components: {e}")
        ai_service = None

    # Initialize database
    await init_db()
    
    yield
    
    # Cleanup
    logger.info("Shutting down application")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=settings.PROJECT_DESCRIPTION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(events.router, prefix="/events", tags=["Events"])
app.include_router(playbooks.router, prefix="/playbooks", tags=["Playbooks"])
app.include_router(system.router, prefix="/system", tags=["System"])
app.include_router(collectors.router)

@app.get("/")
async def root():
    return {
        "message": "AthalaSIEM API is running",
        "ai_service_status": "running" if ai_service else "not initialized"
    }

@app.get("/ai/status")
async def ai_status():
    """Get AI service status"""
    if ai_service:
        return await ai_service.get_service_status()
    return {
        "status": "not_initialized",
        "error": "AI service not initialized"
    }

@app.post("/ai/analyze")
async def analyze_event(event_data: Dict[str, Any]):
    """Analyze event using AI service"""
    if not ai_service:
        return {
            "status": "error",
            "error": "AI service not initialized"
        }
    return await ai_service.analyze_event(event_data)