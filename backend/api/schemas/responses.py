from pydantic import BaseModel
from typing import Optional, Any, TypeVar, Generic
from datetime import datetime

T = TypeVar('T')

class BaseResponse(BaseModel):
    status: str = "success"
    message: Optional[str] = None
    timestamp: datetime = datetime.utcnow()

class DataResponse(BaseResponse, Generic[T]):
    data: T

class ErrorResponse(BaseResponse):
    status: str = "error"
    error_code: str
    detail: str

class PaginatedResponse(DataResponse, Generic[T]):
    total: int
    page: int
    page_size: int
    next_page: Optional[int]
    previous_page: Optional[int]