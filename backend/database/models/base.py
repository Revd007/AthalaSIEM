from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import Column, String, Integer, DateTime, Boolean, func

# Create base class for models
mapper_registry = registry()
Base = mapper_registry.generate_base()