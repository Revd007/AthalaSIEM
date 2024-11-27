import asyncio
import aiodns
import socket
from datetime import datetime
from typing import Dict, Any, Optional

class NetworkCollector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resolver = aiodns.DNSResolver()
        self.buffer_size = self.config.get('buffer_size', 65535)

    async def start_collection(self):
        # Create UDP socket for syslog
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', 514))
        
        while True:
            data, addr = sock.recvfrom(self.buffer_size)
            event = await self.parse_syslog(data, addr)
            yield event

    async def parse_syslog(self, data: bytes, addr: tuple) -> Dict[str, Any]:
        try:
            message = data.decode('utf-8')
            return {
                'timestamp': datetime.utcnow(),
                'source': f'{addr[0]}:{addr[1]}',
                'event_type': 'syslog',
                'message': message,
                'raw_data': data.hex(),
                'severity': self.determine_severity(message)
            }
        except Exception as e:
            print(f"Error parsing syslog message: {e}")
            return None

    def determine_severity(self, message: str) -> int:
        message = message.lower()
        
        # Emergency (0) - System is unusable
        if any(word in message for word in ['emergency', 'emerg', 'panic']):
            return 0
            
        # Alert (1) - Action must be taken immediately
        elif any(word in message for word in ['alert', 'critical', 'crit']):
            return 1
            
        # Error (2) - Error conditions
        elif any(word in message for word in ['error', 'err', 'failed', 'failure']):
            return 2
            
        # Warning (3) - Warning conditions
        elif any(word in message for word in ['warning', 'warn']):
            return 3
            
        # Notice (4) - Normal but significant condition
        elif any(word in message for word in ['notice']):
            return 4
            
        # Info (5) - Informational messages
        elif any(word in message for word in ['info']):
            return 5
            
        # Debug (6) - Debug-level messages
        elif any(word in message for word in ['debug']):
            return 6
            
        # Default to Info level if no keywords match
        return 5
        if any(keyword in message.lower() for keyword in ['error', 'fail', 'critical']):
            return 1
        elif any(keyword in message.lower() for keyword in ['warning', 'warn']):
            return 2
        return 3