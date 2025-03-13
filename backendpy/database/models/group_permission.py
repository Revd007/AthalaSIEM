from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from database.connection import Base

class GroupPermission(Base):
    __tablename__ = "group_permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey('groups.id', ondelete='CASCADE'))
    permission_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    group = relationship("Group", back_populates="permissions")