"""
Auth — Phase 7.4b
Password hashing, JWT creation/verification, and FastAPI dependencies
for protecting routes.

Design:
- Passwords hashed with bcrypt (via passlib)
- Sessions are JWT tokens stored in HttpOnly cookies
- Token expiry: 7 days
- `current_user` dependency raises 401 if not authenticated
- `optional_user` dependency returns None if not authenticated
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, Request, Response
from jose import jwt, JWTError


from .db import get_user_by_id


# ---------- Configuration ----------
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7
SESSION_COOKIE_NAME = "contract_session"

def _get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret or secret.startswith("change_this"):
        raise RuntimeError(
            "JWT_SECRET_KEY is not set (or is still the placeholder). "
            "Edit .env and set a strong random value."
        )
    return secret


# ---------- Password hashing ----------
# ---------- Password hashing (direct bcrypt, no passlib) ----------
# Why direct bcrypt: passlib is unmaintained (last release 2020) and
# incompatible with bcrypt >= 4.1. Bcrypt's 72-byte limit is a known
# security design choice — we truncate explicitly.
import bcrypt


_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    pw_bytes = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    pw_bytes = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))
    except Exception:
        return False


# ---------- JWT helpers ----------
def create_session_token(user_id: int, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXPIRY_DAYS)).timestamp()),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


# ---------- Cookie helpers ----------
def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=JWT_EXPIRY_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=False,   # flip to True when deployed behind HTTPS
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


# ---------- FastAPI dependencies ----------
def optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """Return the user dict if authenticated, else None. Never raises."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None
    try:
        uid = int(payload["sub"])
    except (KeyError, ValueError):
        return None
    user = get_user_by_id(uid)
    return user


def current_user(request: Request) -> Dict[str, Any]:
    """FastAPI dependency: require authentication. Raises 401 if missing."""
    user = optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user