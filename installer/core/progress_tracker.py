from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime
import json
from pathlib import Path

class InstallationProgressTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.steps = {
            'init': 'Initializing installation...',
            'sql_server': 'Installing SQL Server Express...',
            'database': 'Configuring database...',
            'app_files': 'Installing application files...',
            'service': 'Installing Windows service...',
            'cleanup': 'Cleaning up...'
        }
        self.current_step = 'init'
        self.progress = 0
        self.status = 'running'
        self._callbacks = []
        
    def update_progress(self, 
                       step: str, 
                       progress: int, 
                       status: str = 'running',
                       message: Optional[str] = None) -> None:
        """Update installation progress"""
        self.current_step = step
        self.progress = progress
        self.status = status
        
        update = {
            'step': step,
            'step_name': self.steps.get(step, ''),
            'progress': progress,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Notify callbacks
        for callback in self._callbacks:
            callback(update)
            
        # Log progress
        self.logger.info(f"Installation progress: {progress}% - {self.steps.get(step, '')}")