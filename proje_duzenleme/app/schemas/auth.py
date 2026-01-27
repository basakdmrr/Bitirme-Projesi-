
from pydantic import BaseModel

class UserRegister(BaseModel):
    tc: str
    name: str
    password: str

class UserLogin(BaseModel):
    tc: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"