from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/siem_db')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()