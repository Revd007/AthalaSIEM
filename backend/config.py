from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional

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
    # Database
    SIEM_DB_URL: str
    DATABASE_URL: Optional[str] = None
    
    # JWT Settings
    JWT_SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    
    # AI Settings
    ai: AISettings = AISettings()
    
    # API Settings
    API_VERSION: str
    SECRET_KEY: str
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
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
        env_nested_delimiter = '__'
        extra = "allow"

settings = Settings()