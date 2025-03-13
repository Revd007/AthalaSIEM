from pydantic import BaseModel, EmailStr
import uuid

class UserLogin(BaseModel):
    email: EmailStr
    password: str

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
    full_name: str | None = None

    class Config:
        from_attributes = True
        json_encoders = {
            uuid.UUID: str
        }