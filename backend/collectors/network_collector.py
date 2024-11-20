import asyncio
import aiodns
import socket
from datetime import datetime
from typing import Dict, Any

class NetworkCollector:
    def __init__(self):
        self.resolver = aiodns.DNSResolver()
        self.buffer_size = 65535

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
        # Implement severity determination logic
        if any(keyword in message.lower() for keyword in ['error', 'fail', 'critical']):
            return 1
        elif any(keyword in message.lower() for keyword in ['warning', 'warn']):
            return 2
        return 3