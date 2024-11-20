import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "SIEM Backend"
    PROJECT_VERSION: str = "1.0.0"
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database settings
    DB_USER: str = os.getenv("DB_USER", "revian_dbsiem")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "Wokolcoy@20")
    DB_HOST: str = os.getenv("DB_HOST", ".\SQLEXPRESS")
    DB_PORT: str = os.getenv("DB_PORT", "1433")
    DB_NAME: str = os.getenv("DB_NAME", "siem_db")
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
settings = Settings()