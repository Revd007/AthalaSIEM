from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import timedelta
import logging
from sqlalchemy.sql import text

from auth.dependencies.auth_handler import AuthHandler
from auth.schemas.user import UserLogin
from auth.schemas.token import Token
from database.connection import get_db
from auth.utils.password import hash_password
from database.models import User as UserModel
from schemas.user import UserCreate, UserResponse, UserRole
from auth.schemas.auth import LoginRequest, LoginResponse, RegisterRequest
from auth.utils.password import verify_password
from auth.utils.security import create_jwt
from sqlalchemy.sql import func
from config import settings  # Import settings

router = APIRouter()
auth_handler = AuthHandler(secret_key=settings.SECRET_KEY)
logger = logging.getLogger(__name__)

async def get_or_create_default_group(db: AsyncSession):
    from database.models import Group
    
    try:
        query = select(Group).where(Group.name == "Default")
        result = await db.execute(query)
        default_group = result.scalar_one_or_none()
        
        if not default_group:
            default_group = Group(
                name="Default",
                description="Default group for new users"
            )
            db.add(default_group)
            await db.commit()
            await db.refresh(default_group)
        
        return default_group
    except Exception as e:
        logger.error(f"Error creating default group: {str(e)}")
        await db.rollback()
        raise

@router.post("/register", response_model=UserResponse)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    async with db.begin():  # Use transaction
        try:
            # Check existing user
            query = select(UserModel).where(
                or_(
                    UserModel.username == request.username,
                    UserModel.email == request.email
                )
            )

            result = await db.execute(query)
            existing_user = result.scalar_one_or_none()

            if existing_user:
                if existing_user.email == request.email:
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
            hashed_password = hash_password(request.password)
            role_str = request.role.lower()
            user_role = UserRole(role_str)
            
            new_user = UserModel(
                username=request.username,
                email=request.email,
                password_hash=hashed_password,
                full_name=request.full_name,
                role=user_role
            )
            
            db.add(new_user)
            await db.flush()  # Flush to get the user ID
            
            # Get or create default group
            default_group = await get_or_create_default_group(db)
            
            # Assign user to default group using raw SQL
            await db.execute(
                text("""
                    INSERT INTO dbo.user_groups (user_id, group_id, created_at) 
                    VALUES (:user_id, :group_id, CURRENT_TIMESTAMP)
                """),
                {
                    "user_id": new_user.id,
                    "group_id": default_group.id
                }
            )
            
            await db.refresh(new_user)
            
            return UserResponse(
                id=str(new_user.id),
                username=new_user.username,
                email=new_user.email,
                role=new_user.role,
                full_name=new_user.full_name
            )
            
        except Exception as e:
            logger.error(f"Register error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

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
) -> UserResponse:
    """Get current authenticated user"""
    return UserResponse(
        id=str(current_user.id),  # Konversi UUID ke string
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        full_name=current_user.full_name
    )

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