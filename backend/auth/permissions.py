from fastapi import HTTPException, Depends
from functools import wraps
from database.enums import UserRole
from typing import List, Callable
from database.models.user import User

def check_permissions(required_roles: List[UserRole]):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, current_user: User, **kwargs):
            if not any(UserRole.has_permission(current_user.role, role.value) for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to perform this action"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

class Permissions:
    @staticmethod
    def can_manage_users(user: User) -> bool:
        return user.role == UserRole.ADMIN.value

    @staticmethod
    def can_view_analytics(user: User) -> bool:
        return user.role in [UserRole.ADMIN.value, UserRole.ANALYST.value]

    @staticmethod
    def can_manage_alerts(user: User) -> bool:
        return user.role in [UserRole.ADMIN.value, UserRole.ANALYST.value, UserRole.OPERATOR.value]

    @staticmethod
    def can_view_dashboard(user: User) -> bool:
        return True  # All roles can view dashboard