from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from typing import Optional
import logging

class SSLMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app,
        ssl_enabled: bool = True,
        ssl_redirect: bool = True,
        ssl_host: Optional[str] = None
    ):
        super().__init__(app)
        self.ssl_enabled = ssl_enabled
        self.ssl_redirect = ssl_redirect
        self.ssl_host = ssl_host
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next):
        if self.ssl_enabled and self.ssl_redirect:
            if request.url.scheme == "http":
                # Get the host from request or use configured SSL host
                host = self.ssl_host or request.url.hostname
                port = request.url.port or 443

                # Build HTTPS URL
                url = request.url.replace(
                    scheme="https",
                    netloc=f"{host}:{port}" if port != 443 else host
                )
                
                self.logger.info(f"Redirecting to HTTPS: {url}")
                return RedirectResponse(url=str(url))

        try:
            response = await call_next(request)
            
            # Add security headers
            if self.ssl_enabled:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in SSL middleware: {e}")
            raise 