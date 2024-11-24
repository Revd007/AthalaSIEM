from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from auth.dependencies.auth_bearer import JWTBearer
from database.models import User
from auth.utils.security import create_jwt, decode_jwt
from auth.utils.password import verify_password
from database.connection import get_db

class AuthHandler:
    def __init__(self):
        self.db = None

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

    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None):
        return create_jwt(
            data={"user_id": str(user.id), "role": user.role.value},
            expires_delta=expires_delta
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
    
    def decode_token(self, token: str) -> Optional[dict]:
        try:
            return decode_jwt(token)
        except:
            raise HTTPException(status_code=401, detail="Invalid token")