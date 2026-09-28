"""
Phase 1 test — ingestion pipeline (loader + cleaner).
Run: python scripts/test_ingestion.py
"""
import sys
from pathlib import Path

# Make `src` importable regardless of where we run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_document
from src.ingestion.cleaner import clean_document, clean_text


def test_dirty_text():
    print("=" * 60)
    print("TEST 1: Cleaner on deliberately dirty text")
    print("=" * 60)

    dirty = """CONFIDENTIAL

This insur-
ance policy covers the \ufb01re damage.

Page 1 of 3

CONFIDENTIAL

The premi\u00a0um is due     on the 1st.

Page 2 of 3
"""
    cleaned = clean_text(dirty)
    print("Cleaned output:")
    print(cleaned)
    print()
    print("Checks:")
    print("  Hyphenation fixed:", "insurance" in cleaned)
    print("  Ligature fixed:   ", "fire" in cleaned)
    print("  NBSP removed:     ", "\u00a0" not in cleaned)
    print("  Multi-space fixed:", "     " not in cleaned)
    print()


def test_real_document():
    print("=" * 60)
    print("TEST 2: Full pipeline on sample contract")
    print("=" * 60)

    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_contract.txt"
    if not path.exists():
        print(f"SKIP: {path} not found")
        return

    doc = load_document(path)
    print(f"Loaded: format={doc.format}, pages={len(doc.pages)}, chars={len(doc.full_text)}")

    cleaned = clean_document(doc)
    print(f"Cleaned: chars={len(cleaned.full_text)}")
    print()
    print("First 400 chars after cleaning:")
    print("-" * 60)
    print(cleaned.full_text[:400])
    print("-" * 60)


if __name__ == "__main__":
    test_dirty_text()
    test_real_document()
    print("\n>>> Phase 1 ingestion tests complete <<<")