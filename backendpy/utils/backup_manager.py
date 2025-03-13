import shutil
import os
from datetime import datetime
import logging
from pathlib import Path
from typing import *
import json

class BackupManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_dir = Path(config.get('backup_dir', 'backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    async def create_backup(self, backup_type: str = 'full') -> Optional[str]:
        """Create a new backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"backup_{backup_type}_{timestamp}"
            
            if backup_type == 'full':
                return await self._create_full_backup(backup_path)
            elif backup_type == 'config':
                return await self._backup_config(backup_path)
            elif backup_type == 'database':
                return await self._backup_database(backup_path)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
                
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return None
            
    async def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup"""
        try:
            backup = Path(backup_path)
            if not backup.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
                
            # Stop service before restore
            from utils.service_control import ServiceController
            service = ServiceController("AthalaSIEM")
            service.stop()
            
            # Restore files
            if backup.is_file():  # Database backup
                await self._restore_database(backup)
            else:  # Full or config backup
                await self._restore_files(backup)
                
            # Start service
            service.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False