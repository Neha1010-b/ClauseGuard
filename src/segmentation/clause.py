"""
Clause data model — Phase 2.1
The structured unit that segmentation produces and every downstream phase consumes.

Design notes:
- `char_start` / `char_end` refer to positions in the CLEANED text,
  so Phase 8 can highlight by slicing the original.
- `level` encodes hierarchy: "1.1.2" -> 3, "Article IV" -> 1, headings -> 1.
- `parent_id` links nested clauses (1.1 belongs to 1).
- `structural_role` is a lightweight label (HEADING, CLAUSE, RECITAL, etc.)
  — full semantic typing happens in Phase 3 via the classifier.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# Structural roles — distinct from semantic types (Termination, etc.)
ROLE_PREAMBLE = "PREAMBLE"          # "This Agreement..."
ROLE_PARTIES = "PARTIES"            # "BY AND BETWEEN..."
ROLE_RECITAL = "RECITAL"            # "WHEREAS..."
ROLE_HEADING = "HEADING"            # "1. TERMS" (no body, just a title)
ROLE_CLAUSE = "CLAUSE"              # "1.1. The Consultant shall..."
ROLE_SUBCLAUSE = "SUBCLAUSE"        # "a. The Consultant shall..."
ROLE_SIGNATURE = "SIGNATURE"        # "IN WITNESS WHEREOF..."
ROLE_UNKNOWN = "UNKNOWN"


@dataclass
class Clause:
    id: int
    text: str
    char_start: int
    char_end: int
    level: int = 1
    parent_id: Optional[int] = None
    heading: Optional[str] = None          # e.g., "Terms and conditions of Engagement"
    number: Optional[str] = None           # e.g., "1.1" or "(a)" or "IV"
    structural_role: str = ROLE_UNKNOWN
    child_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def length(self) -> int:
        return self.char_end - self.char_start