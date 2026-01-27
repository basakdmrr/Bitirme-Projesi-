from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.user import User
#from app.core.security import get_password_hash, verify_password, create_access_token
from app.schemas.auth import UserRegister, UserLogin, Token
#from app.security.auth import get_current_user
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)

router = APIRouter()

@router.post("/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.tc == user_data.tc).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    
    # Şifreyi hash'le
    hashed_password = get_password_hash(user_data.password)
    
    # User oluştur
    new_user = User(
        tc=user_data.tc, 
        name=user_data.name, 
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = create_access_token(data={"sub": new_user.tc})
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.tc == user_data.tc).first()
    # Doğru field adını kullan: user.hashed_password
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    token = create_access_token(data={"sub": user.tc})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/profile")
def read_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "tc": current_user.tc,
        "name": current_user.name,
        "created_at": current_user.created_at
    }
@router.get("/dashboard")
def get_dashboard(current_user: User = Depends(get_current_user)):
    """Sadece giriş yapmış kullanıcılar görebilir"""
    return {
        "message": f"Hoşgeldin {current_user.name}!",
        "user_id": current_user.id,
        "dashboard_data": "Bu kısım sadece giriş yapmış kullanıcılara özel"
    }

@router.put("/profile")
def update_profile(
    name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Kullanıcı kendi profilini güncelleyebilir"""
    current_user.name = name
    db.commit()
    db.refresh(current_user)
    
    return {"message": "Profil güncellendi", "user": current_user.name}

@router.delete("/account")
def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Kullanıcı kendi hesabını silebilir"""
    db.delete(current_user)
    db.commit()
    return {"message": "Hesap başarıyla silindi"}

@router.get("/secret")
def secret_endpoint(current_user: User = Depends(get_current_user)):
    """Gizli endpoint - sadece giriş yapmış kullanıcılar"""
    return {
        "secret_data": "Bu çok gizli bilgi!",
        "user": current_user.name,
        "message": "Sadece sen görebilirsin"
    }