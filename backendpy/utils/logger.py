import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class Logger:
    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for general logs
        general_handler = RotatingFileHandler(
            f"{log_dir}/siem.log",
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        general_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(general_handler)
        
        # File handler for error logs
        error_handler = RotatingFileHandler(
            f"{log_dir}/error.log",
            maxBytes=10485760,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(error_handler)

    def info(self, message: str, extra: dict = None):
        self.logger.info(message, extra=extra)

    def error(self, message: str, exc_info=True, extra: dict = None):
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def warning(self, message: str, extra: dict = None):
        self.logger.warning(message, extra=extra)

    def debug(self, message: str, extra: dict = None):
        self.logger.debug(message, extra=extra)