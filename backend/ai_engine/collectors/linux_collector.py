import asyncio
import re
from datetime import datetime
from typing import Dict, AsyncGenerator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import logging
import os

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
            '/var/log/messages': 'messages',
            '/var/log/auth.log': 'auth',
            '/var/log/secure': 'secure',
            '/var/log/kern.log': 'kernel',
            '/var/log/dmesg': 'dmesg',
            '/var/log/apache2/access.log': 'apache_access',
            '/var/log/apache2/error.log': 'apache_error',
            '/var/log/httpd/access_log': 'apache_access',
            '/var/log/httpd/error_log': 'apache_error',
            '/var/log/mysql/error.log': 'mysql_error',
            '/var/log/postgresql/postgresql.log': 'postgresql',
            '/var/log/nginx/access.log': 'nginx_access',
            '/var/log/nginx/error.log': 'nginx_error',
            '/var/log/fail2ban.log': 'fail2ban',
            '/var/log/ufw.log': 'firewall',
            '/var/log/audit/audit.log': 'audit',
            '/var/log/cups/error_log': 'cups',
            '/var/log/cron': 'cron',
            '/var/log/maillog': 'mail',
            '/var/log/mail.log': 'mail'
        }
        self.observer = Observer()
        self.handler = LogFileHandler(self._handle_file_change)
        self.active_log_paths = {}
        self._validate_log_paths()
        
    def _validate_log_paths(self):
        """Validasi path log yang ada dan bisa diakses."""
        
        for path, log_type in self.log_paths.items():
            try:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    self.active_log_paths[path] = log_type
                else:
                    logging.info(f"Log path {path} tidak ditemukan atau tidak bisa diakses")
            except Exception as e:
                logging.warning(f"Error saat memeriksa path {path}: {e}")

    def _handle_file_change(self, file_path):
        # Handle file changes here
        pass

    async def collect_logs(self) -> AsyncGenerator[Dict, None]:
        if not self.active_log_paths:
            logging.warning("Tidak ada log path yang aktif untuk dimonitor")
            return

        # Set up watchdog observers hanya untuk file yang ada
        for log_path in self.active_log_paths:
            try:
                self.observer.schedule(self.handler, path=os.path.dirname(log_path), recursive=False)
            except Exception as e:
                logging.error(f"Error setting up observer untuk {log_path}: {e}")

        self.observer.start()

        # Monitor hanya file yang aktif
        file_positions = {}
        for log_path in self.active_log_paths:
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(0, 2)  # Seek ke akhir file
                    file_positions[log_path] = f.tell()
            except Exception as e:
                logging.error(f"Error mendapatkan posisi awal untuk {log_path}: {e}")
                continue  # Skip file yang bermasalah

        # Baca log dari file yang aktif
        for log_path, log_type in self.active_log_paths.items():
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    while True:
                        try:
                            line = f.readline()
                            if not line:
                                await asyncio.sleep(0.1)
                                continue

                            event = self.parse_log_line(line, log_type)
                            if event:
                                yield event
                        except UnicodeDecodeError as e:
                            logging.warning(f"Unicode decode error di {log_path}: {e}")
                            continue
            except Exception as e:
                logging.error(f"Error membaca {log_path}: {e}")
                continue

    def parse_log_line(self, line: str, log_type: str) -> Dict:
        timestamp_patterns = [
            r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{4})',
            r'(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})',
        ]
        
        try:
            for pattern in timestamp_patterns:
                timestamp_match = re.search(pattern, line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        for fmt in [
                            '%b %d %H:%M:%S',
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S%z',
                            '%d/%b/%Y:%H:%M:%S'
                        ]:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        return {
                            'timestamp': timestamp,
                            'source': log_type,
                            'event_type': 'linux_log',
                            'message': line.strip(),
                            'raw_data': line,
                            'severity': self.determine_severity(line),
                            'host': self._get_hostname(),
                            'facility': self._determine_facility(line),
                            'process_id': self._extract_pid(line)
                        }
                    except Exception as e:
                        logging.debug(f"Failed to parse timestamp {timestamp_str}: {e}")
            
            return {
                'timestamp': datetime.now(),
                'source': log_type,
                'event_type': 'linux_log',
                'message': line.strip(),
                'raw_data': line,
                'severity': self.determine_severity(line),
                'host': self._get_hostname(),
                'facility': self._determine_facility(line),
                'process_id': self._extract_pid(line)
            }
            
        except Exception as e:
            logging.error(f"Error parsing log line: {e}")
            return None

    def _get_hostname(self) -> str:
        try:
            import socket
            return socket.gethostname()
        except:
            return "unknown"

    def _determine_facility(self, message: str) -> str:
        facilities = {
            'kern': r'kernel:|kern\.',
            'user': r'user\.',
            'mail': r'mail\.|postfix|sendmail',
            'daemon': r'daemon\.|systemd',
            'auth': r'auth\.|sshd|sudo|su:',
            'syslog': r'syslog\.',
            'cron': r'CRON|cron\.',
            'security': r'security\.|fail2ban|ufw',
        }
        
        for facility, pattern in facilities.items():
            if re.search(pattern, message, re.IGNORECASE):
                return facility
        return "other"

    def _extract_pid(self, message: str) -> str:
        pid_match = re.search(r'\[(\d+)\]', message)
        return pid_match.group(1) if pid_match else "unknown"

    def determine_severity(self, message: str) -> int:
        message = message.lower()
        
        if any(word in message for word in ['emergency', 'emerg', 'panic', 'alert']):
            return 0
        elif any(word in message for word in ['critical', 'crit', 'fatal']):
            return 1
        elif any(word in message for word in ['error', 'err', 'failed', 'failure']):
            return 2
        elif any(word in message for word in ['warning', 'warn', 'could not']):
            return 3
        elif any(word in message for word in ['notice', 'info']):
            return 4
        elif 'debug' in message:
            return 5
        return 4
    