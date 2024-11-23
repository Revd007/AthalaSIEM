from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import yaml
from pathlib import Path
import torch

from api.middleware.cors import setup_cors
from api.middleware.auth import AuthHandler
from api.middleware.error_handler import global_error_handler
from api.middleware.logging import RequestLoggingMiddleware
from database.connection import DatabaseManager
from api.routes import alerts, events, users, playbooks, system
from config import settings
from ai_engine.core.model_manager import ModelManager
from ai_engine.core.logger_config import setup_logger

# Setup logging
logging.basicConfig(
    handlers=[RotatingFileHandler(
        'logs/siem.log', 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Setup AI logger
ai_logger = setup_logger()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI with middleware
middleware = [
    Middleware(SlowAPIMiddleware, limiter=limiter)
]

app = FastAPI(
    title="SIEM API", 
    version="1.0.0",
    middleware=middleware
)

# Add rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"}
    )

# Initialize AI components with proper error handling
try:
    ai_config_path = Path("backend/ai_engine/services/ai_settings.yaml")
    if not ai_config_path.exists():
        ai_logger.warning("AI config file not found, using default settings")
        ai_config = {
            'threat_detector': {
                'input_dim': 512,
                'embedding_dim': 768,
                'hidden_dim': 256,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1
            },
            'anomaly_detector': {
                'input_dim': 256,
                'hidden_dims': [128, 64],
                'latent_dim': 32
            },
            'use_gpu': True
        }
    else:
        with open(ai_config_path) as f:
            ai_config = yaml.safe_load(f)
    
    # Initialize model manager with correct configuration
    model_manager = ModelManager(config=ai_config)
except Exception as e:
    ai_logger.error(f"Failed to initialize AI components: {e}")
    model_manager = None

# Setup middleware
setup_cors(app)
app.middleware("http")(RequestLoggingMiddleware())

# Initialize database with error handling
try:
    db = DatabaseManager(settings.DATABASE_URL)
except Exception as e:
    logging.error(f"Database initialization error: {e}")
    raise

# Setup authentication
auth_handler = AuthHandler(settings.SECRET_KEY)

# Mengakses AI settings
training_interval = settings.ai.TRAINING_INTERVAL
batch_size = settings.ai.BATCH_SIZE
enable_auto_training = settings.ai.ENABLE_AUTO_TRAINING

# Mengakses settings umum
database_url = settings.DATABASE_URL
api_version = settings.API_VERSION

async def start_auto_training():
    """Start automatic training cycle"""
    if model_manager:
        ai_logger.info("Starting automatic training system")
        try:
            await model_manager.auto_train_and_evaluate(
                data_interval=settings.TRAINING_INTERVAL
            )
        except Exception as e:
            ai_logger.error(f"Error in auto training: {e}")
    else:
        ai_logger.warning("Auto-training disabled: Model manager not initialized")

@app.on_event("startup")
async def startup_event():
    """Startup events when application starts"""
    try:
        # Start auto training in background if model manager is available
        if model_manager:
            ai_logger.info("Initializing auto-training system")
            asyncio.create_task(start_auto_training())
        
        ai_logger.info("Application startup completed")
    except Exception as e:
        ai_logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        ai_logger.info("Shutting down application")
        # Cleanup code here if needed
    except Exception as e:
        ai_logger.error(f"Shutdown error: {e}")

# Add routes with rate limiting
@app.get("/")
async def root(request: Request):
    return {"status": "ok", "message": "SIEM API is running"}

# Include routers with rate limiting
app.include_router(
    alerts.router,
    prefix="/api/alerts",
    tags=["alerts"]
)

app.include_router(
    events.router,
    prefix="/api/events",
    tags=["events"]
)

app.include_router(
    users.router,
    prefix="/api/users",
    tags=["users"]
)

app.include_router(
    playbooks.router,
    prefix="/api/playbooks",
    tags=["playbooks"]
)

app.include_router(
    system.router,
    prefix="/api/system",
    tags=["system"]
)

# Add global error handler
app.add_exception_handler(Exception, global_error_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)