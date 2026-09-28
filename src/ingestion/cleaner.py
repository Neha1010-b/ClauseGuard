"""
Text Cleaner — Phase 1.2
Normalizes raw extracted text into a canonical form suitable for segmentation.

Key operations:
1. Unicode normalization (NFKC) — ligatures, compatibility chars
2. Remove repeated header/footer lines (per-page detection)
3. Fix hyphenation across line breaks
4. Collapse whitespace (spaces, tabs, non-breaking spaces)
5. Preserve paragraph structure (blank lines between sections)
"""
import re
import unicodedata
from collections import Counter
from typing import List, Optional

from .loader import LoadedDocument, PageText


# ----------------------------------------------------------------
# Individual cleaning primitives
# ----------------------------------------------------------------

# Ligatures that PDF extraction often leaves in
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft",
}


def normalize_unicode(text: str) -> str:
    """NFKC normalization + ligature replacement."""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    return unicodedata.normalize("NFKC", text)


def remove_hyphenation(text: str) -> str:
    """
    Join words that were split across line breaks: 'insur-\\nance' -> 'insurance'.
    Only when hyphen is at end of line AND next line starts with a lowercase letter.
    """
    return re.sub(r"(\w+)-\n([a-z])", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """
    Collapse runs of spaces/tabs into one space.
    Handle non-breaking spaces intelligently:
      - Between two letters  -> remove entirely (was an intra-word NBSP)
      - Elsewhere            -> convert to regular space
    Preserve single newlines.
    """
    # Step 1: intra-word NBSP -> delete (join the two letters)
    # \w matches [A-Za-z0-9_]; we also want to catch accented letters, so use a broader pattern
    text = re.sub(r"(?<=\w)[\u00A0\u2007\u202F\u2009\u200A](?=\w)", "", text)

    # Step 2: remaining exotic whitespace -> regular space
    text = re.sub(r"[\u00A0\u2007\u202F\u2009\u200A]", " ", text)

    # Step 3: multiple spaces/tabs -> single space
    text = re.sub(r"[ \t]+", " ", text)

    # Step 4: space before newline -> remove
    text = re.sub(r" +\n", "\n", text)

    # Step 5: 3+ newlines -> 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_repeated_lines(pages: List[PageText], min_page_fraction: float = 0.5) -> set:
    """
    Find lines that repeat across many pages — these are almost always
    headers/footers. Returns a set of line strings to remove.
    
    min_page_fraction: a line must appear on at least this fraction of pages
    to be considered a header/footer.
    """
    if len(pages) < 2:
        return set()

    # Normalize each line for comparison: strip + collapse internal spaces
    def norm(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    # Count pages where each normalized line appears
    line_page_counts: Counter = Counter()
    for page in pages:
        # Use a set so a line counted once per page, even if repeated within page
        unique_lines = {norm(l) for l in page.text.splitlines() if l.strip()}
        for line in unique_lines:
            line_page_counts[line] += 1

    threshold = len(pages) * min_page_fraction
    return {line for line, count in line_page_counts.items() if count >= threshold and len(line) > 2}


def strip_repeated_lines(text: str, repeated: set) -> str:
    """Remove any line whose normalized form is in `repeated`."""
    def norm(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    kept = [line for line in text.splitlines() if norm(line) not in repeated]
    return "\n".join(kept)


# ----------------------------------------------------------------
# Page-number and footer patterns (regex fallbacks)
# ----------------------------------------------------------------
_FOOTER_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),                        # bare page numbers
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),         # - 3 -
    re.compile(r"^\s*CONFIDENTIAL\s*$", re.IGNORECASE),
    re.compile(r"^\s*DRAFT\s*$", re.IGNORECASE),
]


def strip_footer_patterns(text: str) -> str:
    """Remove lines matching known page-number/footer conventions."""
    kept = []
    for line in text.splitlines():
        if any(p.match(line) for p in _FOOTER_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------
def clean_text(text: str, pages: Optional[List[PageText]] = None) -> str:
    """
    Full cleaning pipeline. Order matters:
    1. Unicode + ligatures  (before any pattern matching)
    2. Hyphenation          (before whitespace collapse)
    3. Repeated line strip  (needs page info)
    4. Footer patterns      (after unicode so "Page 3" is clean ASCII)
    5. Whitespace collapse  (last — normalizes the result)
    """
    text = normalize_unicode(text)
    text = remove_hyphenation(text)

    if pages:
        repeated = detect_repeated_lines(pages)
        text = strip_repeated_lines(text, repeated)

    text = strip_footer_patterns(text)
    text = collapse_whitespace(text)
    return text


def clean_document(doc: LoadedDocument) -> LoadedDocument:
    """Apply cleaning to a LoadedDocument, returning a new one with cleaned text."""
    cleaned = clean_text(doc.full_text, pages=doc.pages)
    # Rebuild pages with cleaned full text but preserve metadata.
    # For a single-page doc, this is trivial; for multi-page, we rebuild offsets.
    new_pages: List[PageText] = []
    cursor = 0
    for p in doc.pages:
        cleaned_page = clean_text(p.text)
        if not cleaned_page:
            continue
        new_pages.append(PageText(
            page_number=p.page_number,
            text=cleaned_page,
            char_start=cursor,
            char_end=cursor + len(cleaned_page),
        ))
        cursor += len(cleaned_page)

    return LoadedDocument(
        source_path=doc.source_path,
        format=doc.format,
        full_text=cleaned,
        pages=new_pages,
        metadata={**doc.metadata, "cleaned": True},
    )