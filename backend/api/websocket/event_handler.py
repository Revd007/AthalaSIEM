from fastapi import WebSocket
from typing import Dict, Set
import json
import logging
from datetime import datetime

class EventHandler:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "alerts": set(),
            "events": set(),
            "system": set()
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].add(websocket)
            logging.info(f"Client connected to {channel} channel")

    async def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            self.active_connections[channel].remove(websocket)
            logging.info(f"Client disconnected from {channel} channel")

    async def broadcast(self, channel: str, message: dict):
        if channel not in self.active_connections:
            return

        message["timestamp"] = datetime.utcnow().isoformat()
        
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Failed to send message: {e}")
                await self.disconnect(connection, channel)