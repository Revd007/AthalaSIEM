from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging

async def global_error_handler(request: Request, exc: Exception):
    logging.error(f"Global error: {exc}")
    
    if hasattr(exc, 'status_code'):
        status_code = exc.status_code
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": str(exc),
            "path": request.url.path
        }
    )