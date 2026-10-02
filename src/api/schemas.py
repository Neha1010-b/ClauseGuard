"""
Pydantic schemas — Phase 7.2
Defines the exact shape of API requests and responses.

These schemas are the contract between backend and frontend.
FastAPI uses them for:
  - Automatic request validation
  - Automatic response serialization
  - Auto-generated OpenAPI docs at /docs
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------- Response sub-models ----------

class DocumentInfo(BaseModel):
    filename: str
    format: str
    pages: int
    chars: int


class SummaryStats(BaseModel):
    total_clauses: int
    high: int
    medium: int
    low: int
    skipped_structural: int


class ClassificationInfo(BaseModel):
    label: Optional[str] = None
    confidence: float = 0.0
    top_k: List[Dict[str, Any]] = Field(default_factory=list)


class RiskInfo(BaseModel):
    score: float
    level: str                                  # "low" | "medium" | "high"
    categories: List[str] = Field(default_factory=list)
    explanation_hint: str = ""
    signals: Dict[str, Any] = Field(default_factory=dict)


class ExplanationInfo(BaseModel):
    text: Optional[str] = None
    suggested_action: Optional[str] = None
    model_used: Optional[str] = None
    latency_ms: int = 0
    skipped_reason: Optional[str] = None


class ClauseResult(BaseModel):
    id: int
    number: Optional[str] = None
    structural_role: str
    level: int
    parent_id: Optional[int] = None
    text: str
    char_start: int
    char_end: int
    classification: ClassificationInfo
    risk: RiskInfo
    explanation: ExplanationInfo


class AnalysisResponse(BaseModel):
    document: DocumentInfo
    summary: SummaryStats
    clauses: List[ClauseResult]


# ---------- Error response ----------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ---------- Auth schemas ----------

class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6, max_length=200)


class SigninRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=1, max_length=200)


class AuthUserResponse(BaseModel):
    id: int
    email: str
    full_name: str