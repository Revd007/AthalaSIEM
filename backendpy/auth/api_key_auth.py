from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
from datetime import datetime
import hashlib
import hmac

class APIKeyAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
        
    async def validate_api_key(
        self, 
        api_key: str = Security(APIKeyHeader(name="X-API-Key"))
    ) -> bool:
        try:
            # Validate API key format
            if not self._is_valid_format(api_key):
                return False
                
            # Extract timestamp and signature
            timestamp, signature = api_key.split(".")
            
            # Check if timestamp is within valid window (5 minutes)
            if not self._is_timestamp_valid(timestamp):
                return False
                
            # Verify signature
            expected_signature = self._generate_signature(timestamp)
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
            
    def _is_valid_format(self, api_key: str) -> bool:
        parts = api_key.split(".")
        return len(parts) == 2 and all(p for p in parts)
        
    def _is_timestamp_valid(self, timestamp: str) -> bool:
        try:
            ts = int(timestamp)
            now = int(datetime.utcnow().timestamp())
            return abs(now - ts) <= 300  # 5 minutes
        except ValueError:
            return False
            
    def _generate_signature(self, timestamp: str) -> str:
        message = f"{timestamp}.{self.secret_key}"
        return hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest() 