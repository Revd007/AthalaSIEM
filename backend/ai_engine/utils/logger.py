import logging
import sys
from typing import Optional
from datetime import datetime
import os
import json
from pathlib import Path

class AILogger:
    def __init__(self, 
                 name: str = "AI_Engine",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Training history
        self.training_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"ai_engine_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_training_step(self, 
                         epoch: int,
                         metrics: dict,
                         model_name: Optional[str] = None):
        """Log training step with metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            'model_name': model_name
        }
        
        # Add to history
        self.training_history.append(log_entry)
        
        # Log to file
        self.logger.info(
            f"Training Step - Epoch {epoch} - "
            f"Model: {model_name or 'Unknown'} - "
            f"Metrics: {json.dumps(metrics, indent=2)}"
        )
        
        # Save history
        self._save_training_history()
    
    def log_model_version(self,
                         version: str,
                         changes: dict,
                         performance: dict):
        """Log model version update"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'version': version,
            'changes': changes,
            'performance': performance
        }
        
        self.logger.info(
            f"Model Version Update - Version: {version}\n"
            f"Changes: {json.dumps(changes, indent=2)}\n"
            f"Performance: {json.dumps(performance, indent=2)}"
        )
        
        # Save to version history file
        version_file = self.log_dir / 'version_history.jsonl'
        with open(version_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_error(self, error: Exception, context: Optional[dict] = None):
        """Log error with context"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        self.logger.error(
            f"Error: {error_entry['error_type']} - "
            f"{error_entry['error_message']}\n"
            f"Context: {json.dumps(context or {}, indent=2)}"
        )
        
        # Save to error log file
        error_file = self.log_dir / 'error_log.jsonl'
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
    
    def _save_training_history(self):
        """Save training history to file"""
        history_file = self.log_dir / 'training_history.jsonl'
        with open(history_file, 'w') as f:
            for entry in self.training_history:
                f.write(json.dumps(entry) + '\n')