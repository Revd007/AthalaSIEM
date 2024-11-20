from api.routes.events import router as events_router
from api.routes.alerts import router as alerts_router
from api.routes.dashboard import router as dashboard_router
from auth.routes.auth import router as auth_router

__all__ = ['events_router', 'alerts_router', 'dashboard_router', 'auth_router']