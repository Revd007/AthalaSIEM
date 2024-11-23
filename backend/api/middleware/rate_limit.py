from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis
import time

class RateLimiter:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.rate_limits = {
            "default": (100, 60),  # 100 requests per minute
            "auth": (5, 60),       # 5 login attempts per minute
            "api": (1000, 3600)    # 1000 requests per hour for API endpoints
        }

    async def check_rate_limit(
        self, 
        request: Request, 
        limit_type: str = "default"
    ) -> bool:
        client_ip = get_remote_address(request)
        key = f"rate_limit:{limit_type}:{client_ip}"
        
        # Get rate limit configuration
        max_requests, window = self.rate_limits.get(
            limit_type, 
            self.rate_limits["default"]
        )
        
        # Get current count
        current = self.redis.get(key)
        
        if current is None:
            # First request, set initial count
            self.redis.setex(key, window, 1)
            return True
            
        current = int(current)
        
        if current >= max_requests:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
            
        # Increment request count
        self.redis.incr(key)
        return True