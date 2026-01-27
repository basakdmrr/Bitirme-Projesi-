from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Numeric
from sqlalchemy.sql import func
from app.db.database import Base

class VoiceRecord(Base):
    __tablename__ = "voice_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tc = Column(String(11), nullable=True)  # <-- tc alanı eklendi
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    prediction_result = Column(String, nullable=True)
    confidence_score = Column(Numeric(5, 4), nullable=True)
    processing_status = Column(String(50), nullable=False, server_default="pending")