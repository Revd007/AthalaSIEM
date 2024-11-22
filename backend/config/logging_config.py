import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging"""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'logs/siem.log')
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_config.get('max_size', 10485760),  # 10MB
                backupCount=log_config.get('backup_count', 5)
            ),
            logging.StreamHandler()  # Console output
        ]
    )
    
    # Configure specific loggers
    loggers = {
        'uvicorn': log_level,
        'sqlalchemy': logging.WARNING,
        'aiohttp': logging.WARNING,
        'asyncio': logging.WARNING
    }
    
    for logger_name, logger_level in loggers.items():
        logging.getLogger(logger_name).setLevel(logger_level)