from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jwt import encode, decode
from auth.dependencies.auth_bearer import JWTBearer
from database.models import User
from auth.utils.security import create_jwt, decode_jwt
from auth.utils.password import verify_password
from database.connection import get_db

class AuthHandler:
    security = HTTPBearer()
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = 'HS256'
        
    def encode_token(self, user_id: str) -> str:
        payload = {
            'exp': datetime.utcnow() + timedelta(hours=8),
            'iat': datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(
            payload,
            self.secret_key,
            algorithm='HS256'
        )
        
    def decode_token(self, token: str) -> Dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail='Token has expired')
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail='Invalid token')
            
    def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Security(security)):
        return self.decode_token(auth.credentials)

    async def authenticate_user(
        self, 
        username: str, 
        password: str, 
        db: AsyncSession = Depends(get_db)
    ) -> Optional[User]:
        query = select(User).filter(User.username == username)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(password, user.password_hash):
            return None
        return user

    def create_access_token(
        self, 
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        return encode(
            payload=to_encode,
            key=self.secret_key,
            algorithm=self.algorithm
        )

    async def get_current_user(
        self, 
        token: str = Depends(JWTBearer()),
        db: AsyncSession = Depends(get_db)
    ) -> User:
        payload = decode_jwt(token)
        query = select(User).filter(User.id == payload["user_id"])
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user