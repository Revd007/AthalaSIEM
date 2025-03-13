from fastapi import Request, HTTPException
import re
from typing import List, Dict

class RequestValidator:
    def __init__(self):
        # Regex patterns untuk validasi
        self.patterns = {
            'sql_injection': r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            'xss': r"(<script|javascript:|vbscript:|<img|<iframe)",
            'path_traversal': r"(\.\.\/|\.\.\\)",
            'command_injection': r"(;|\||`|\$\(|\))"
        }
        
    async def validate_request(self, request: Request):
        # Validasi path parameters
        path_params = request.path_params
        for param in path_params.values():
            await self._validate_input(str(param))
            
        # Validasi query parameters    
        query_params = request.query_params
        for param in query_params.values():
            await self._validate_input(str(param))
            
        # Validasi body jika ada
        if request.method in ['POST', 'PUT', 'PATCH']:
            body = await request.json()
            await self._validate_json(body)
            
    async def _validate_input(self, value: str):
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input detected: {pattern_name}"
                )
                
    async def _validate_json(self, data: Dict):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    await self._validate_input(value)
                elif isinstance(value, (dict, list)):
                    await self._validate_json(value)
        elif isinstance(data, list):
            for item in data:
                await self._validate_json(item) 