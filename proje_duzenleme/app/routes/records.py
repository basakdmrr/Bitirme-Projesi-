from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.voice_record import VoiceRecord
from app.security.auth import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/")
def get_my_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    records = (
        db.query(VoiceRecord)
        .filter(VoiceRecord.user_id == current_user.id)
        .order_by(VoiceRecord.created_at.desc())
        .all()
    )

    return records
