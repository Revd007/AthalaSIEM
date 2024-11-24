from datetime import datetime, timedelta
from typing import Optional
import jwt
from jwt import encode, decode
from database.settings import settings

def create_jwt(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    return encode(
        payload=to_encode,
        key=settings.SECRET_KEY,
        algorithm="HS256"
    )

def decode_jwt(token: str) -> dict:
    return decode(
        jwt=token,
        key=settings.SECRET_KEY,
        algorithms=["HS256"]
    )