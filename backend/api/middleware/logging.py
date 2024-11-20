from fastapi import Request
import time
import logging
from typing import Callable
import json

class RequestLoggingMiddleware:
    async def __call__(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        logging.info(f"Request started: {request.method} {request.url}")
        
        try:
            # Get request body
            body = await request.body()
            if body:
                logging.debug(f"Request body: {body.decode()}")
        except Exception as e:
            logging.error(f"Error reading request body: {e}")

        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logging.info(
            f"Request completed: {request.method} {request.url} "
            f"Status: {response.status_code} "
            f"Processing time: {process_time:.3f}s"
        )
        
        return response