from sqlalchemy import Column, String, DateTime, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database.connection import Base
from database.enums import UserRole
from .group import user_groups

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole, name='user_role', create_type=False, 
                      values_callable=lambda obj: [e.value for e in obj]), 
                 nullable=False, 
                 default=UserRole.VIEWER.value)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    groups = relationship("Group", secondary=user_groups, back_populates="users")
    created_groups = relationship("Group", back_populates="creator", foreign_keys="[Group.created_by]")

    @property
    def permissions(self):
        base_permissions = {
            UserRole.VIEWER.value: ["view_dashboard", "view_alerts"],
            UserRole.OPERATOR.value: ["view_dashboard", "view_alerts", "acknowledge_alerts", "update_alerts"],
            UserRole.ANALYST.value: ["view_dashboard", "view_alerts", "acknowledge_alerts", "update_alerts", 
                                   "create_reports", "view_analytics"],
            UserRole.ADMIN.value: ["view_dashboard", "view_alerts", "acknowledge_alerts", "update_alerts",
                                 "create_reports", "view_analytics", "manage_users", "manage_groups",
                                 "manage_settings", "manage_playbooks"]
        }
        return base_permissions.get(self.role, [])

    def __repr__(self):
        return f"<User {self.username}>"

    @property
    def is_authenticated(self):
        return True

    @property
    def is_staff(self):
        return self.role in [UserRole.ADMIN.value, UserRole.ANALYST.value, UserRole.OPERATOR.value]

    @property
    def can_manage_users(self):
        return self.role == UserRole.ADMIN.value

    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "full_name": self.full_name
        }