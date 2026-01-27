from pydantic import BaseModel
from datetime import datetime

class UserBase(BaseModel):
    tc: str
    name: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    tc: str
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True  