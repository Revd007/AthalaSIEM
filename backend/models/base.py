from pydantic import BaseModel

class BaseConfig(BaseModel):
    class Config:
        from_attributes = True  # Ganti orm_mode menjadi from_attributes