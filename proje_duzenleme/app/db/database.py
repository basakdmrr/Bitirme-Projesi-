from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL bulunamadı! .env dosyasını kontrol edin.")

print(f"Bağlantı URL: {DATABASE_URL}")

# Base sınıfı
Base = declarative_base()

# MSSQL bağlantısı için engine
engine = create_engine(DATABASE_URL, echo=True)

# Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()