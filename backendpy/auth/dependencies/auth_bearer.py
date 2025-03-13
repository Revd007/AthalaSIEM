from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from ..utils.security import decode_jwt

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials = await super(JWTBearer, self).__call__(request)
        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization token")
        
        token = credentials.credentials
        if not self.verify_jwt(token):
            raise HTTPException(status_code=403, detail="Invalid token or expired token")
            
        return token

    def verify_jwt(self, token: str) -> bool:
        try:
            payload = decode_jwt(token)
            return True if payload else False
        except:
            return False