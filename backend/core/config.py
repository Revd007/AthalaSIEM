from pydantic import BaseSettings, Field
from typing import Dict, Any, Optional
from pathlib import Path
import os

class AISettings(BaseSettings):
    # Training intervals
    TRAINING_INTERVAL: int = Field(default=3600, description="Training interval in seconds")
    EVALUATION_INTERVAL: int = Field(default=7200, description="Evaluation interval in seconds")
    
    # Model parameters
    BATCH_SIZE: int = Field(default=32, description="Training batch size")
    LEARNING_RATE: float = Field(default=2e-5, description="Model learning rate")
    MAX_EPOCHS: int = Field(default=10, description="Maximum training epochs")
    EARLY_STOPPING_PATIENCE: int = Field(default=3, description="Epochs to wait before early stopping")
    
    # Paths
    MODEL_CHECKPOINT_DIR: str = Field(
        default="checkpoints",
        description="Directory to save model checkpoints"
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for AI logs"
    )
    
    # Performance thresholds
    MIN_ACCURACY: float = Field(default=0.8, description="Minimum accuracy threshold")
    MIN_F1_SCORE: float = Field(default=0.75, description="Minimum F1 score threshold")
    
    # Resource limits
    MAX_MEMORY_USAGE: float = Field(
        default=0.8,
        description="Maximum memory usage threshold (0-1)"
    )
    MAX_GPU_MEMORY: float = Field(
        default=0.9,
        description="Maximum GPU memory usage threshold (0-1)"
    )
    
    # Feature toggles
    ENABLE_AUTO_TRAINING: bool = Field(
        default=True,
        description="Enable automatic training"
    )
    ENABLE_GPU: bool = Field(
        default=True,
        description="Enable GPU usage if available"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = Field(
        default="mssql+aioodbc://user:password@localhost:1433/siem_db?driver=ODBC+Driver+17+for+SQL+Server",
        description="Database connection string"
    )
    
    # API settings
    API_VERSION: str = Field(default="1.0.0", description="API version")
    SECRET_KEY: str = Field(default="your-secret-key", description="JWT secret key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # AI settings
    ai: AISettings = AISettings()
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # Security
    ALLOWED_HOSTS: list = Field(default=["*"], description="Allowed hosts")
    CORS_ORIGINS: list = Field(default=["*"], description="Allowed CORS origins")
    
    # System
    ENVIRONMENT: str = Field(default="development", description="Environment (development/production)")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create necessary directories
        os.makedirs(self.ai.MODEL_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.ai.LOG_DIR, exist_ok=True)

settings = Settings()