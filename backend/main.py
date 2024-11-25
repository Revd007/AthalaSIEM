from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth.routes import auth
from api.routes import alerts, events, users, playbooks, system, dashboard
from database.connection import init_db
from database.models.user import UserRole
from config import settings
import logging
from pathlib import Path
import yaml
from ai_engine.donquixote_service import DonquixoteService

# Setup logging
logging.basicConfig(level=logging.INFO)
ai_logger = logging.getLogger('ai_engine')

# Initialize FastAPI app
app = FastAPI(title="AthalaSIEM API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components with proper error handling
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

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(events.router, prefix="/api/v1", tags=["events"])
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
app.include_router(users.router, prefix="/api/v1", tags=["users"])
app.include_router(playbooks.router, prefix="/api/v1", tags=["playbooks"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])

@app.on_event("startup")
async def startup():
    # Initialize database
    await init_db()

@app.get("/")
async def root():
    return {"message": "AthalaSIEM API is running"}