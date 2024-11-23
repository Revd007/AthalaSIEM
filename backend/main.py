from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import alerts, events, users, playbooks, system
from config import settings
from database.connection import init_db

# Initialize FastAPI app
app = FastAPI(
    title="SIEM API",
    version=settings.API_VERSION,
    description="SIEM System API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    await init_db()

# Include routers
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(playbooks.router, prefix="/api/playbooks", tags=["playbooks"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)