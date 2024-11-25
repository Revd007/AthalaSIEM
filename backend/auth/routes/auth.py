from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import timedelta
import logging
from sqlalchemy.sql import or_

from auth.dependencies.auth_handler import AuthHandler
from auth.schemas.user import UserLogin
from auth.schemas.token import Token
from database.connection import get_db
from auth.utils.password import hash_password
from database.models import User as UserModel
from schemas.user import UserCreate, UserResponse, UserRole
from auth.schemas.auth import LoginRequest, LoginResponse
from auth.utils.password import verify_password
from auth.utils.security import create_jwt
from sqlalchemy.sql import func
from config import settings  # Import settings

router = APIRouter()
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)
logger = logging.getLogger(__name__)

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        # Check if user exists
        query = select(UserModel).filter(
            or_(
                UserModel.email == user.email,
                UserModel.username == user.username
            )
        )
        result = await db.execute(query)
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            if existing_user.email == user.email:
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Username already taken"
                )
        
        # Create new user
        hashed_password = hash_password(user.password)
        db_user = UserModel(
            email=user.email,
            username=user.username,
            password_hash=hashed_password,
            full_name=user.full_name,
            role=UserRole.VIEWER
        )
        
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        return UserResponse(
            id=str(db_user.id),
            email=db_user.email,
            username=db_user.username,
            role=db_user.role,
            full_name=db_user.full_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during registration"
        )

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        logger.info(f"Login attempt for username: {login_data.username}")
        # Find user by username
        query = select(UserModel).filter(UserModel.username == login_data.username)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Verify password
        if not verify_password(login_data.password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Update last login
        user.last_login = func.now()
        await db.commit()
        
        # Generate JWT token
        access_token = create_jwt(
            data={
                "sub": str(user.id),
                "username": user.username,
                "role": user.role
            },
            expires_delta=timedelta(minutes=30)
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during login"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: UserModel = Depends(auth_handler.get_current_user)
):
    return current_user

@router.post("/logout")
async def logout(current_user: UserModel = Depends(auth_handler.get_current_user)):
    return {"message": "Successfully logged out"}

@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    current_user: UserModel = Depends(auth_handler.get_current_user),
):
    access_token = auth_handler.create_access_token(
        current_user,
        expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}