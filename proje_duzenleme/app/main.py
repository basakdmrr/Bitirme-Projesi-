from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import Base, engine
from app.routes import auth, predict, records

# DB tablolarını oluştur
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Voice Disease Detection API",
    description="Ses verilerinden hastalık tespiti yapan FastAPI backend",
    version="1.0.0"
)

# CORS (mobil / frontend için şart)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production'da domain ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ları ekle
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(records.router, prefix="/records", tags=["Records"])

# Health check
@app.get("/", tags=["Health"])
def root():
    return {"status": "API çalışıyor 🚀"}
