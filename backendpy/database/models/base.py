from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import Column, String, Integer, DateTime, Boolean, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Create base class for models
mapper_registry = registry()
Base = mapper_registry.generate_base()

# Gunakan UUID dari PostgreSQL sebagai pengganti UNIQUEIDENTIFIER
# Contoh penggunaan dalam model:
id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)