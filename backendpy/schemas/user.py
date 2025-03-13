from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from database.enums import UserRole

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = Field(default=UserRole.VIEWER)
    is_active: bool = True

    class Config:
        from_attributes = True
        use_enum_values = True
        json_encoders = {
            UserRole: lambda v: v.value
        }

class UserCreate(UserBase):
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: UserRole
    full_name: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {
            UserRole: lambda v: v.value
        }

# Export all models
__all__ = ['UserRole', 'UserBase', 'UserCreate', 'UserResponse']