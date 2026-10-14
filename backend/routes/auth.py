import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from config import settings

logger = logging.getLogger("auth_router")
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str


class UserProfileResponse(BaseModel):
    username: str
    role: str
    permissions: list


@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    NOTE: This is a demo-only static login (admin/password123), not real auth.
    Fine for a project demo; mention this limitation if asked in your review.
    """
    try:
        admin_username = "admin"
        admin_password = "password123"

        if form_data.username != admin_username or form_data.password != admin_password:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Incorrect username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        return TokenResponse(access_token=settings.SECRET_KEY, expires_at=expires_at.isoformat())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Authentication server error.")


@router.get("/me", response_model=UserProfileResponse, status_code=status.HTTP_200_OK)
async def read_current_profile(token: str = Depends(oauth2_scheme)):
    try:
        if token != settings.SECRET_KEY:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Invalid token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return UserProfileResponse(
            username="admin",
            role="Administrator",
            permissions=["all_access", "read_attendance", "write_students"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile read error: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Profile lookup failure.")
