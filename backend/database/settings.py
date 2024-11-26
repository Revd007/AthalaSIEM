import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    # Project settings
    PROJECT_NAME: str = "AthalaSIEM"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "Security Information and Event Management System"
    AI_MODEL_NAME: str = "Donquixote Athala"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fae1044cf8b3c10cbe8c50933f090633593372e61db672db0873e27cc96438c3")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database settings
    DB_USER: str = os.getenv("DB_USER", "revian_dbsiem")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "wokolcoy020")
    DB_HOST: str = os.getenv("DB_HOST", "46.250.234.160")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "siem_db")
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3000"
        "http://localhost:8080",
    ]
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
    # Service settings
    PORT: int = int(os.getenv("PORT", "8080"))
    USE_HTTPS: bool = os.getenv("USE_HTTPS", "false").lower() == "true"
    SSL_KEYFILE: str = os.getenv("SSL_KEYFILE", "certs/key.pem")
    SSL_CERTFILE: str = os.getenv("SSL_CERTFILE", "certs/cert.pem")
    
    # Database settings
    DB_TYPE: str = os.getenv("DB_TYPE", "POSTGRESQL")
    DB_AUTO_INSTALL: bool = True
    
    # Service management
    SERVICE_NAME: str = "AthalaSIEM"
    SERVICE_DISPLAY_NAME: str = "AthalaSIEM Service"
    SERVICE_DESCRIPTION: str = "AthalaSIEM Security Information and Event Management Service"
    
    class Config:
        from_attributes = True

settings = Settings()