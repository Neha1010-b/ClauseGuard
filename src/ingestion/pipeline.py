"""
Unified ingestion entry point.
Public API: ingest(path) -> LoadedDocument (fully loaded and cleaned).
This is the ONLY function Phase 2 (segmentation) needs to call.
"""
from pathlib import Path
from typing import Union

from .loader import load_document, LoadedDocument
from .cleaner import clean_document


def ingest(path: Union[str, Path], validate: bool = True) -> LoadedDocument:
    """
    Load + clean a contract file. One call, one result.

    Args:
        path: Path to PDF / DOCX / TXT file.
        validate: If True, reject documents whose cleaned text is too short
                  (likely scanned PDF or broken extraction).

    Returns:
        LoadedDocument with cleaned text + page offsets + metadata.

    Raises:
        FileNotFoundError, ValueError, RuntimeError
    """
    doc = load_document(path)
    cleaned = clean_document(doc)

    if validate:
        from ..utils.config import get_config
        min_len = get_config()["ingestion"]["min_text_length"]
        if len(cleaned.full_text) < min_len:
            raise RuntimeError(
                f"Document too short after cleaning: {len(cleaned.full_text)} chars "
                f"(minimum {min_len}). File may be a scanned PDF requiring OCR, "
                f"or extraction may have failed."
            )

    # Attach a small provenance record
    cleaned.metadata["ingestion"] = {
        "source_format": doc.format,
        "original_chars": len(doc.full_text),
        "cleaned_chars": len(cleaned.full_text),
        "page_count": len(cleaned.pages),
    }
    return cleaned