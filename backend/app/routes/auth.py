import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from models.database import User, get_db
from models.schemas import UserRegister, UserLogin, Token, UserResponse
from services.auth import hash_password, verify_password, create_access_token, get_current_user, require_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token)
async def register_user(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check if email exists
    existing_user = db.query(User).filter(User.email == user_in.email.lower()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists."
        )

    hashed_pwd = hash_password(user_in.password)
    new_user = User(
        email=user_in.email.lower(),
        name=user_in.name or user_in.email.split("@")[0],
        hashed_password=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(data={"sub": new_user.id})
    user_resp = UserResponse.model_validate(new_user)
    return Token(access_token=token, token_type="bearer", user=user_resp)


@router.post("/login", response_model=Token)
async def login_user(user_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email.lower()).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    token = create_access_token(data={"sub": user.id})
    user_resp = UserResponse.model_validate(user)
    return Token(access_token=token, token_type="bearer", user=user_resp)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(require_current_user)):
    return UserResponse.model_validate(current_user)
