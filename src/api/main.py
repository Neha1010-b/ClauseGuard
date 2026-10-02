"""
FastAPI application — Phase 7.3
Exposes the risk analyzer as a REST API.

Endpoints:
    GET  /              - health check + basic info
    GET  /health        - detailed health check
    POST /analyze       - upload a contract, get full analysis
    POST /analyze/quick - same as /analyze but skips LLM explanations

Run (dev):
    uvicorn src.api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.
"""
import os
import uuid
import time
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.config import get_config, PROJECT_ROOT
from .schemas import AnalysisResponse, ErrorResponse
from .pipeline import analyze_document

from fastapi import Request, Response, Depends
from . import db as db_module
from . import auth as auth_module
from .schemas import (
    SignupRequest, SigninRequest, AuthUserResponse,
)


# ============================================================
# App setup
# ============================================================
app = FastAPI(
    title="Contract Clause Risk Analyzer",
    description=(
        "Upload a rental agreement, employment contract, or vendor agreement. "
        "The system extracts clauses, classifies them, scores risk, and "
        "generates plain-English explanations for the risky ones."
    ),
    version="0.7.0",
)

# CORS — allow the frontend (which we'll serve from a different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # For production, restrict to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config once
_cfg = get_config()
_api_cfg = _cfg["api"]
_ing_cfg = _cfg["ingestion"]
_MAX_UPLOAD_MB = int(_api_cfg.get("max_upload_size_mb", 20))
_SUPPORTED_FORMATS = set(_ing_cfg.get("supported_formats", ["pdf", "docx", "txt"]))

# Where to put temporary uploads
_TMP_DIR = PROJECT_ROOT / "data" / "raw" / "_tmp_uploads"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper — validate + save uploaded file
# ============================================================
def _save_upload(upload: UploadFile) -> Path:
    """
    Save the uploaded file to a temp location and return the path.
    Raises HTTPException on invalid format or size.
    """
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(upload.filename).suffix.lower().lstrip(".")
    if ext not in _SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '.{ext}'. Supported: {sorted(_SUPPORTED_FORMATS)}",
        )

    # Write with a unique prefix to avoid collisions
    unique = uuid.uuid4().hex[:12]
    dest = _TMP_DIR / f"{unique}_{upload.filename}"

    # Save in chunks and enforce max size
    max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        with open(dest, "wb") as f:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {_MAX_UPLOAD_MB} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    return dest


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "name": "Contract Clause Risk Analyzer",
        "version": app.version,
        "status": "ok",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "POST /analyze",
            "analyze_quick": "POST /analyze/quick",
        },
    }


@app.get("/health")
def health():
    """
    Detailed health check. Verifies that all models and config are loaded.
    Useful for deployment monitoring.
    """
    checks = {}

    # Config
    try:
        checks["config"] = "ok"
    except Exception as e:
        checks["config"] = f"error: {e}"

    # Classifier model files
    model_path = PROJECT_ROOT / _cfg["classification"]["model_save_path"]
    checks["classifier_model"] = "ok" if model_path.exists() else f"missing: {model_path}"

    # FAISS index
    index_path = PROJECT_ROOT / _cfg["reference_bank"]["index_path"]
    checks["faiss_index"] = "ok" if index_path.exists() else f"missing: {index_path}"

    # Gemini API key
    checks["gemini_api_key"] = "ok" if os.getenv("GEMINI_API_KEY") else "missing GEMINI_API_KEY"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT contract file"),
    generate_explanations: bool = Form(True),
):
    """
    Full analysis: ingest → segment → classify → compare → score → explain.

    If generate_explanations=False, the LLM step is skipped (faster, no API cost).
    """
    start = time.time()
    dest = _save_upload(file)
    try:
        result = analyze_document(
            dest,
            generate_explanations=generate_explanations,
            show_progress=False,
        )
        elapsed = time.time() - start
        print(f"[api] analyzed {file.filename} in {elapsed:.1f}s — "
              f"high={result.summary.high} medium={result.summary.medium} low={result.summary.low}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(e).__name__}: {str(e)[:200]}",
        )
    finally:
        dest.unlink(missing_ok=True)


@app.post("/analyze/quick", response_model=AnalysisResponse)
def analyze_quick(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT contract file"),
):
    """
    Same as /analyze but skips LLM explanations.
    Faster and free — useful for testing the pipeline.
    """
    return analyze(file, generate_explanations=False)


# ============================================================
# Error handler — always return JSON, never HTML
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)[:300]},
    )

# ============================================================
# Auth endpoints
# ============================================================
@app.on_event("startup")
def _startup_init_db():
    """Ensure DB tables exist on every server start."""
    db_module.init_db()


@app.post("/auth/signup", response_model=AuthUserResponse)
def signup(payload: SignupRequest, response: Response):
    """Create a new account and log the user in."""
    email = payload.email.lower().strip()
    if db_module.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered")

    password_hash = auth_module.hash_password(payload.password)
    try:
        user_id = db_module.create_user(email, payload.full_name, password_hash)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")

    token = auth_module.create_session_token(user_id, email)
    auth_module.set_session_cookie(response, token)

    user = db_module.get_user_by_id(user_id)
    return AuthUserResponse(id=user["id"], email=user["email"], full_name=user["full_name"])


@app.post("/auth/signin", response_model=AuthUserResponse)
def signin(payload: SigninRequest, response: Response):
    """Log in with email + password."""
    email = payload.email.lower().strip()
    user = db_module.get_user_by_email(email)
    if not user or not auth_module.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = auth_module.create_session_token(user["id"], user["email"])
    auth_module.set_session_cookie(response, token)

    return AuthUserResponse(id=user["id"], email=user["email"], full_name=user["full_name"])


@app.post("/auth/signout")
def signout(response: Response):
    """Clear the session cookie."""
    auth_module.clear_session_cookie(response)
    return {"ok": True}


@app.get("/auth/me", response_model=AuthUserResponse)
def me(user: dict = Depends(auth_module.current_user)):
    """Return the current user's profile."""
    return AuthUserResponse(id=user["id"], email=user["email"], full_name=user["full_name"])