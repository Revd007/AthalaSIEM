from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from auth.models.user import UserRole

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: str
    role: UserRole
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True