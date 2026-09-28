"""
Phase 2 test — boundary detection on the real contract.
Run: python scripts/test_segmentation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation.segmenter import segment


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"

    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 60)
    print(f"Loading and segmenting: {pdf_path.name}")
    print("=" * 60)

    doc = ingest(pdf_path)
    print(f"Total chars: {len(doc.full_text)}")
    print()

    clauses = segment(doc.full_text)
    print()
    print(f"Total clauses detected: {len(clauses)}")
    print()
    print("First 15 clauses (number | role | first 70 chars):")
    print("-" * 60)
    for c in clauses[:15]:
        num = c.number or "—"
        preview = c.text[:70].replace("\n", " ")
        print(f"  #{c.id:02d} [{num:>6}] {c.structural_role:>10} | {preview}")


if __name__ == "__main__":
    main()