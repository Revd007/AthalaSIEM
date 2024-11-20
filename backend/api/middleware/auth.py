from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional
import logging

class JWTAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.security = HTTPBearer()

    def create_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode = {
            "exp": expire,
            "user_id": user_id
        }
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await self.security(request)
        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
        
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=["HS256"])
            user_id = payload.get("user_id")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError as e:
            logging.error(f"JWT validation error: {e}")
            raise HTTPException(status_code=401, detail="Could not validate credentials")