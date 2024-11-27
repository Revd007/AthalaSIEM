from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import json
import logging
from .base_collector import BaseCollector

class MacOSCollector(BaseCollector):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_collecting = False
        self.log_paths = {
            'system': '/var/log/system.log',
            'security': '/var/log/security.log',
            'install': '/var/log/install.log'
        }

    async def start_collection(self):
        """Start collecting logs from MacOS sources"""
        try:
            self.is_collecting = True
            self.logger.info("MacOS log collection started")
        except Exception as e:
            self.logger.error(f"Error starting MacOS collection: {e}")
            self.is_collecting = False

    async def stop_collection(self):
        """Stop collecting logs"""
        try:
            self.is_collecting = False
            self.logger.info("MacOS log collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping MacOS collection: {e}")

    async def get_logs(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get collected logs within time range and filters"""
        try:
            logs = []
            for log_type, log_path in self.log_paths.items():
                # Build log command based on filters
                cmd = ['log', 'show']
                
                if start_time:
                    cmd.extend(['--start', start_time.strftime('%Y-%m-%d %H:%M:%S')])
                if end_time:
                    cmd.extend(['--end', end_time.strftime('%Y-%m-%d %H:%M:%S')])
                
                # Add predicate if filters are provided
                if filters:
                    predicate = self._build_predicate(filters)
                    if predicate:
                        cmd.extend(['--predicate', predicate])
                
                # Execute command and parse output
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        parsed_logs = self._parse_logs(result.stdout, log_type)
                        logs.extend(parsed_logs)
                except subprocess.SubprocessError as e:
                    self.logger.error(f"Error executing log command: {e}")
                    continue

            return logs
        except Exception as e:
            self.logger.error(f"Error getting MacOS logs: {e}")
            return []

    async def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        return {
            'active': self.is_collecting,
            'log_paths': self.log_paths,
            'collector_type': 'macos'
        }

    def _build_predicate(self, filters: Dict[str, Any]) -> str:
        """Build log show predicate from filters"""
        predicates = []
        
        if 'process' in filters:
            predicates.append(f'process == "{filters["process"]}"')
        if 'category' in filters:
            predicates.append(f'category == "{filters["category"]}"')
        if 'level' in filters:
            predicates.append(f'eventType == "{filters["level"]}"')
        
        return ' AND '.join(predicates) if predicates else ''

    def _parse_logs(self, log_content: str, log_type: str) -> List[Dict[str, Any]]:
        """Parse log content into structured format"""
        logs = []
        for line in log_content.splitlines():
            try:
                if line.strip():
                    log_entry = {
                        'timestamp': None,
                        'process': None,
                        'message': line,
                        'type': log_type,
                        'raw': line
                    }
                    
                    # Try to extract timestamp and process
                    parts = line.split(None, 3)
                    if len(parts) >= 3:
                        try:
                            log_entry['timestamp'] = ' '.join(parts[:2])
                            log_entry['process'] = parts[2]
                            log_entry['message'] = parts[3] if len(parts) > 3 else ''
                        except Exception:
                            pass
                    
                    logs.append(log_entry)
            except Exception as e:
                self.logger.error(f"Error parsing log line: {e}")
                continue
                
        return logs