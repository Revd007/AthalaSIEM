from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

    @classmethod
    def has_permission(cls, role: str, required_role: str) -> bool:
        role_hierarchy = {
            "admin": ["admin", "analyst", "operator", "viewer"],
            "analyst": ["analyst", "operator", "viewer"],
            "operator": ["operator", "viewer"],
            "viewer": ["viewer"]
        }
        return required_role in role_hierarchy.get(role.lower(), [])