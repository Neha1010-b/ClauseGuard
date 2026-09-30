"""
Clause segmenter — Phase 2.5 (full implementation)

Pipeline:
  normalized text
    -> detect candidate boundaries (may overlap)
    -> resolve overlaps (prefer longer / more specific)
    -> build spans between consecutive boundaries
    -> classify each span's structural role
    -> merge tiny fragments into previous clause
    -> build parent-child hierarchy
    -> return list of Clause objects
"""
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .clause import (
    Clause, ROLE_HEADING, ROLE_CLAUSE, ROLE_SUBCLAUSE, ROLE_PREAMBLE,
    ROLE_RECITAL, ROLE_PARTIES, ROLE_SIGNATURE, ROLE_UNKNOWN,
)
from ..utils.config import get_config


# ============================================================
# Text normalization (for matching only — original preserved)
# ============================================================
_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})


def _normalize_for_matching(text: str, cfg: dict) -> str:
    if cfg.get("normalize_quotes") or cfg.get("normalize_dashes"):
        return text.translate(_QUOTE_MAP)
    return text


# ============================================================
# Boundary detection
# ============================================================
@dataclass
class Boundary:
    position: int          # char index in normalized text
    kind: str              # "number" | "structural"
    number: Optional[str]  # clause number label
    depth: int             # hierarchy depth (from number); 0 for structural
    role: str              # suggested structural role
    specificity: int       # tie-breaker: higher = more specific match
    raw_match: str


def _depth_from_number(num: Optional[str]) -> int:
    """'1.1.2' -> 3; 'ARTICLE IV' -> 1; '(a)' -> 2; None -> 0."""
    if not num:
        return 0
    if "." in num and num[0].isdigit():
        return num.count(".") + 1
    if num.startswith("(") and num.endswith(")"):
        inner = num[1:-1]
        if inner.isalpha() and inner.lower() in {
            "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
        }:
            return 3   # roman sub-items are deeper
        return 2       # (a), (b) are level 2
    if num.startswith("ARTICLE") or num.startswith("Section") or num.startswith("Clause"):
        return 1
    if num and num[0].isalpha():
        return 2
    if num.isdigit():
        return 1
    return 1


def _find_numbered_boundaries(text: str, cfg: dict) -> List[Boundary]:
    """Detect numbered clause markers. May over-fire — dedup later."""
    boundaries: List[Boundary] = []
    for pat_str in cfg["numbering_patterns"]:
        try:
            pattern = re.compile(pat_str)
        except re.error as e:
            print(f"[warn] Invalid pattern {pat_str!r}: {e}")
            continue

        for m in pattern.finditer(text):
            if "num" not in m.groupdict() or not m.group("num"):
                continue
            num = m.group("num")
            # Position the boundary at the start of the number, not the anchor
            num_offset = m.group(0).find(num)
            pos = m.start() + num_offset

            boundaries.append(Boundary(
                position=pos,
                kind="number",
                number=num,
                depth=_depth_from_number(num),
                role=ROLE_CLAUSE,
                specificity=num.count(".") + len(num),
                raw_match=num,
            ))
    return boundaries


def _find_structural_boundaries(text: str, cfg: dict) -> List[Boundary]:
    boundaries: List[Boundary] = []
    for entry in cfg.get("structural_markers", []):
        pattern = re.compile(entry["pattern"], re.MULTILINE)
        for m in pattern.finditer(text):
            boundaries.append(Boundary(
                position=m.start(),
                kind="structural",
                number=None,
                depth=1,
                role=entry["role"],
                specificity=100,
                raw_match=m.group(0),
            ))
    return boundaries


# ============================================================
# Overlap resolution
# ============================================================
def _resolve_overlaps(boundaries: List[Boundary], window: int = 5) -> List[Boundary]:
    """
    If two boundaries are within `window` chars of each other, keep the
    one with higher specificity. Structural boundaries always win.
    """
    if not boundaries:
        return []

    boundaries = sorted(boundaries, key=lambda b: (b.position, -b.specificity))
    kept: List[Boundary] = []
    for b in boundaries:
        if kept and abs(b.position - kept[-1].position) <= window:
            if b.specificity > kept[-1].specificity:
                kept[-1] = b
        else:
            kept.append(b)
    return kept


# ============================================================
# Span construction
# ============================================================
def _build_spans(text: str, boundaries: List[Boundary]) -> List[Tuple[int, int, Boundary]]:
    """Build (start, end, boundary) tuples from consecutive boundaries."""
    spans: List[Tuple[int, int, Boundary]] = []
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1].position if i + 1 < len(boundaries) else len(text)
        spans.append((b.position, end, b))
    return spans


# ============================================================
# Structural role classification
# ============================================================
_HEADING_PATTERN = re.compile(r"^\s*(?:\d+\.|[A-Z][A-Z\s&\-]{3,})\s*$")


def _classify_role(text: str, boundary: Boundary) -> str:
    """Decide what kind of block this span is."""
    stripped = text.strip()
    if not stripped:
        return ROLE_UNKNOWN

    if boundary.kind == "structural":
        return boundary.role

    if boundary.number:
        if boundary.depth >= 2:
            return ROLE_SUBCLAUSE
        return ROLE_CLAUSE

    if _HEADING_PATTERN.match(stripped) and len(stripped) < 80:
        return ROLE_HEADING

    return ROLE_UNKNOWN


# ============================================================
# Fragment merging — offset-preserving
# ============================================================
def _merge_tiny_fragments(
    clauses: List[Clause],
    full_text: str,
    min_chars: int,
    min_words: int,
) -> List[Clause]:
    """
    If a clause is too small, merge it into the previous clause's span.

    EXCEPTION: if the tiny clause looks like a heading (short, capitalized,
    ends with a period or is all-caps), keep it as its own clause. Headings
    are semantically meaningful — merging them pollutes the next clause's
    hierarchy.

    Invariant preserved:
        clause.text == full_text[clause.char_start:clause.char_end]
    """
    if not clauses:
        return []

    import re
    heading_like = re.compile(
        r"^(?:\d+\.\s*)?[A-Z][A-Za-z\s&\-:,]{2,80}\.?$"
    )

    def looks_like_heading(c: Clause) -> bool:
        # Must have a number (e.g., "8. Termination") OR be very short + capitalized
        text = c.text.strip()
        # Length guard — headings are short
        if len(text) > 80:
            return False
        # Must end with . or be all-caps or be title-cased short
        if not (text.endswith(".") or text.isupper() or
                (len(text.split()) <= 6 and text[0].isupper())):
            return False
        return True

    merged: List[Clause] = []
    for c in clauses:
        words = len(c.text.split())
        too_small = c.length < min_chars or words < min_words
        if too_small and merged and not looks_like_heading(c):
            # Swallow this clause's span into previous
            prev = merged[-1]
            prev.char_end = c.char_end
            prev.text = full_text[prev.char_start:prev.char_end]
            continue
        merged.append(c)

    for i, c in enumerate(merged):
        c.id = i

    for c in merged:
        c.parent_id = None
        c.child_ids = []

    return merged


# ============================================================
# Hierarchy building
# ============================================================
def _build_hierarchy(clauses: List[Clause], max_level: int) -> List[Clause]:
    """
    Set parent_id / child_ids based on numbering depth.
    A clause with level N belongs to the nearest preceding clause with level < N.
    """
    stack: List[Clause] = []
    for c in clauses:
        lvl = min(c.level, max_level)
        c.level = lvl
        while stack and stack[-1].level >= lvl:
            stack.pop()
        if stack:
            c.parent_id = stack[-1].id
            stack[-1].child_ids.append(c.id)
        stack.append(c)
    return clauses


# ============================================================
# Public API
# ============================================================
def segment(text: str) -> List[Clause]:
    """
    Segment cleaned contract text into a list of Clause objects.
    Fully deterministic given config — same input, same output.
    """
    cfg = get_config()["segmentation"]
    normalized = _normalize_for_matching(text, cfg)

    # 1. Detect boundaries (may over-fire)
    numbered = _find_numbered_boundaries(normalized, cfg)
    structural = _find_structural_boundaries(normalized, cfg)
    all_b = numbered + structural

    # 2. Resolve overlaps
    resolved = _resolve_overlaps(all_b, window=5)
    resolved.sort(key=lambda b: b.position)

    # 3. Build spans (using resolved positions)
    spans = _build_spans(text, resolved)

    # 4. Construct Clause objects
    #    Note: we set char_start/char_end from raw span positions, then
    #    immediately re-derive text from full_text to guarantee the
    #    text == full_text[start:end] invariant.
    clauses: List[Clause] = []
    for i, (start, end, b) in enumerate(spans):
        raw = text[start:end]
        # Skip leading whitespace so highlighting starts tight on content
        leading_ws = len(raw) - len(raw.lstrip())
        adjusted_start = start + leading_ws
        adjusted_end = end

        # Trim trailing whitespace too
        while adjusted_end > adjusted_start and text[adjusted_end - 1].isspace():
            adjusted_end -= 1

        # Re-derive text — never use `raw.strip()` as the source of truth
        text_slice = text[adjusted_start:adjusted_end]

        clauses.append(Clause(
            id=i,
            text=text_slice,
            char_start=adjusted_start,
            char_end=adjusted_end,
            level=b.depth,
            number=b.number,
            structural_role=_classify_role(text_slice, b),
            metadata={"boundary_kind": b.kind},
        ))

    # 5. Merge tiny fragments (pass full_text so re-slicing works)
    clauses = _merge_tiny_fragments(
        clauses,
        full_text=text,
        min_chars=cfg["min_clause_length"],
        min_words=cfg["min_clause_words"],
    )

    # 6. Build hierarchy (AFTER merging, so IDs are consistent)
    clauses = _build_hierarchy(clauses, max_level=cfg["max_level"])

    # 7. Diagnostics
    print(f"[segmenter] boundaries: {len(numbered)} numeric + "
          f"{len(structural)} structural = {len(all_b)} raw -> "
          f"{len(resolved)} after dedup -> {len(clauses)} clauses")

    return clauses