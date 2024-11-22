from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from datetime import timedelta

from auth.dependencies.auth_handler import AuthHandler
from auth.schemas.user import UserLogin
from auth.schemas.token import Token
from database.connection import get_db
from auth.utils.password import hash_password
from database.models import User as UserModel
from schemas.user import UserCreate, UserResponse, UserRole

router = APIRouter()
auth_handler = AuthHandler()

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(UserModel).filter(UserModel.email == user.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    if db.query(UserModel).filter(UserModel.username == user.username).first():
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
        role=UserRole.VIEWER  # Use the enum directly
    )
    
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating user: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = await auth_handler.authenticate_user(user_data.email, user_data.password, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="User account is disabled"
        )
    
    # Update last login
    user.last_login = func.now()
    await db.commit()
    
    access_token = auth_handler.create_access_token(
        user, expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

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