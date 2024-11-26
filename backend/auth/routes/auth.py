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
        
        # Create new user dengan role yang diinputkan
        hashed_password = hash_password(user.password)
        
        # Pastikan role dalam lowercase dan valid
        role = user.role.lower() if isinstance(user.role, str) else user.role.value.lower()
        
        db_user = UserModel(
            email=user.email,
            username=user.username,
            password_hash=hashed_password,
            full_name=user.full_name,
            role=role  # Gunakan role yang sudah diformat
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
            detail=f"Internal server error during registration: {str(e)}"
        )

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        # Add debug logging
        logger.debug(f"Login attempt for username: {login_data.username}")
        
        query = select(UserModel).filter(UserModel.username == login_data.username)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            logger.warning(f"Login failed: User not found - {login_data.username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        if not verify_password(login_data.password, user.password_hash):
            logger.warning(f"Login failed: Invalid password for user {login_data.username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        # Update last login
        user.last_login = func.now()
        await db.commit()
        
        access_token = create_jwt(
            data={
                "user_id": str(user.id),
                "username": user.username,
                "role": user.role
            }
        )
        
        logger.info(f"Login successful for user {login_data.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "username": user.username,
                "role": user.role,
                "full_name": user.full_name
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during login: {str(e)}"
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