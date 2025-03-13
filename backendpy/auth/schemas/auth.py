from pydantic import BaseModel, EmailStr
from database.enums import UserRole

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    full_name: str | None = None