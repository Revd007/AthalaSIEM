import asyncio
import re
from datetime import datetime
from typing import Dict, AsyncGenerator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import logging

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self.callback(event.src_path)

class LinuxLogCollector:
    def __init__(self):
        self.log_paths = {
            '/var/log/syslog': 'syslog',
            '/var/log/auth.log': 'auth',
            '/var/log/kern.log': 'kernel',
            '/var/log/apache2/access.log': 'apache_access',
            '/var/log/apache2/error.log': 'apache_error'
        }
        self.observer = Observer()
        self.handler = LogFileHandler(self._handle_file_change)
        
    def _handle_file_change(self, file_path):
        # Handle file changes here
        pass

    async def collect_logs(self) -> AsyncGenerator[Dict, None]:
        # Set up watchdog observers for each file
        for log_path in self.log_paths:
            try:
                self.observer.schedule(self.handler, path=log_path, recursive=False)
            except Exception as e:
                logging.error(f"Error setting up observer for {log_path}: {e}")
        
        self.observer.start()

        # Rest of the collect_logs method remains the same
        for log_path, log_type in self.log_paths.items():
            try:
                with open(log_path, 'r') as f:
                    f.seek(0, 2)  # Seek to end of file
                    while True:
                        line = f.readline()
                        if not line:
                            await asyncio.sleep(0.1)
                            continue
                        
                        event = self.parse_log_line(line, log_type)
                        if event:
                            yield event
            except Exception as e:
                logging.error(f"Error reading {log_path}: {e}")

    def parse_log_line(self, line: str, log_type: str) -> Dict:
        timestamp_pattern = r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'
        try:
            timestamp_match = re.search(timestamp_pattern, line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, '%b %d %H:%M:%S')
                
                return {
                    'timestamp': timestamp,
                    'source': log_type,
                    'event_type': 'linux_log',
                    'message': line.strip(),
                    'raw_data': line,
                    'severity': self.determine_severity(line)
                }
        except Exception as e:
            logging.error(f"Error parsing log line: {e}")
            return None

    def determine_severity(self, message: str) -> int:
        if any(keyword in message.lower() for keyword in ['emergency', 'alert', 'critical', 'error']):
            return 1
        elif 'warning' in message.lower():
            return 2
        elif 'notice' in message.lower():
            return 3
        return 4