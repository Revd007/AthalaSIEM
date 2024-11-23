from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

def setup_cors(app: FastAPI, allowed_origins: List[str] = None):
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # Frontend development
            "http://localhost:8000",  # Backend development
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )