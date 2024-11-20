from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

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
        email: str, 
        password: str, 
        db: Session = Depends(get_db)
    ) -> Optional[User]:
        user = db.query(User).filter(User.email == email).first()
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
        db: Session = Depends(get_db)
    ) -> User:
        payload = decode_jwt(token)
        user = db.query(User).filter(User.id == payload["user_id"]).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user