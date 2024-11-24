import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database.connection import AsyncSessionLocal
from database.models.user import User, UserRole
from auth.utils.password import hash_password
import uuid

async def create_admin_user():
    async with AsyncSessionLocal() as session:
        try:
            # Check if admin user already exists
            query = f"SELECT * FROM dbo.users WHERE username = 'admin'"
            result = await session.execute(query)
            admin_user = result.first()
            
            if not admin_user:
                # Create new admin user
                admin_user = User(
                    id=uuid.uuid4(),
                    username='admin',
                    email='admin@athala.com',
                    password_hash=hash_password('admin'),
                    full_name='System Administrator',
                    role=UserRole.ADMIN,
                    is_active=True
                )
                
                session.add(admin_user)
                await session.commit()
                print("Admin user created successfully")
            else:
                print("Admin user already exists")
        except Exception as e:
            print(f"Error creating admin user: {str(e)}")
            await session.rollback()
            raise e

# Run the async function
if __name__ == "__main__":
    asyncio.run(create_admin_user())