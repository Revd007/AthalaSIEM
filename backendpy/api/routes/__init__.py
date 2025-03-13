from .events import router as events_router
from .users import router as users_router
from .playbooks import router as playbooks_router
from .system import router as system_router
from .dashboard import router as dashboard_router
from .collectors import router as collectors_router
from .ai_service import router as ai_router
from .alerts import router as alerts_router

__all__ = [
    'alerts',
    'events',
    'users',
    'playbooks',
    'system',
    'dashboard',
    'collectors',
    'ai_service'
] 