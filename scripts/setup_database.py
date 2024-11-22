# Buat file setup_database.py
from backend.database.models import Base
from backend.database.connection import engine

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()