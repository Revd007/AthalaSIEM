from typing import List
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional
import json

class Config:
    env_file = ".env"

class AISettings(BaseModel):
    TRAINING_INTERVAL: int = 3600
    EVALUATION_INTERVAL: int = 7200
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    MAX_EPOCHS: int = 10
    EARLY_STOPPING_PATIENCE: int = 3
    MODEL_CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    MIN_ACCURACY: float = 0.8
    MIN_F1_SCORE: float = 0.75
    MAX_MEMORY_USAGE: float = 0.8
    MAX_GPU_MEMORY: float = 0.9
    ENABLE_AUTO_TRAINING: bool = True
    ENABLE_GPU: bool = True

class Settings(BaseSettings):
    DATABASE_URL: str = "mssql+pyodbc://revian_dbsiem:Wokolcoy%4020@localhost/siem_db?driver=ODBC+Driver+17+for+SQL+Server"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Settings
    ai: AISettings = AISettings()
    
    # API Settings
    API_VERSION: str
    SECRET_KEY: str
    
    # Logging
    LOG_LEVEL: str
    
    # Security
    ALLOWED_HOSTS: List[str]
    CORS_ORIGINS: List[str]
    
    # System
    ENVIRONMENT: str
    DEBUG: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.DATABASE_URL:
            self.DATABASE_URL = self.SIEM_DB_URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields

# Create settings instance with environment variables
settings = Settings(_env_file='.env', _env_file_encoding='utf-8')
settings = Settings()