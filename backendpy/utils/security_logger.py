import logging
from datetime import datetime
from typing import Dict, Any

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        fh = logging.FileHandler("logs/security.log")
        fh.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(fh)
        
    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ):
        self.logger.log(
            logging.getLevelName(severity),
            f"Security Event: {event_type} - {details}"
        ) 