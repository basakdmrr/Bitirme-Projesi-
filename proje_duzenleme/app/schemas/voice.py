from pydantic import BaseModel
from datetime import datetime

class VoiceRecordCreate(BaseModel):
    tc: str
    result: str
    file_path: str

class VoiceRecordOut(BaseModel):
    id: int
    tc: str
    result: str
    file_path: str
    created_at: datetime

    class Config:
        from_attributes = True