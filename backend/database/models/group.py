from sqlalchemy import Column, String, DateTime, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database.connection import Base

# Many-to-many relationship table for users and groups
user_groups = Table(
    'user_groups',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('public.users.id', ondelete='CASCADE')),
    Column('group_id', UUID(as_uuid=True), ForeignKey('public.groups.id', ondelete='CASCADE')),
    Column('created_at', DateTime, default=func.now()),
    schema='public'
)

class Group(Base):
    __tablename__ = "groups"
    __table_args__ = {"schema": "public"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255))
    created_by = Column(UUID(as_uuid=True), ForeignKey('public.users.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship("User", secondary=user_groups, back_populates="groups")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_groups")
    permissions = relationship("GroupPermission", back_populates="group", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Group {self.name}>"