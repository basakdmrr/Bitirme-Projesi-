from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import os, shutil, uuid

from app.db.database import get_db
from app.models.voice_record import VoiceRecord
from app.core.security import get_current_user
from app.ml.inference import predict_audio

router = APIRouter(tags=["Prediction"])

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 🔴 content_type yerine filename kontrolü
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only WAV files supported")

    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, confidence = predict_audio(file_path)

        record = VoiceRecord(
            user_id=current_user.id,
            tc=current_user.tc,
            file_path=filename,
            prediction_result=label,
            confidence_score=round(confidence, 4),
            processing_status="completed"
        )

        db.add(record)
        db.commit()
        db.refresh(record)

    finally:
        os.remove(file_path)

    return {
        "record_id": record.id,
        "prediction": label,
        "confidence": round(confidence, 4)
    }

