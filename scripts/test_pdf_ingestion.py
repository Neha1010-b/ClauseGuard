"""
Test PDF ingestion path specifically.
Run: python scripts/test_pdf_ingestion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_contract.pdf"

    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found. Create it first.")
        return

    print("=" * 60)
    print(f"Ingesting: {pdf_path.name}")
    print("=" * 60)

    doc = ingest(pdf_path)

    print(f"Format:   {doc.format}")
    print(f"Pages:    {len(doc.pages)}")
    print(f"Chars:    {len(doc.full_text)}")
    print(f"Metadata: {doc.metadata['ingestion']}")
    print()
    print("First 300 chars of cleaned text:")
    print("-" * 60)
    print(doc.full_text[:300])
    print("-" * 60)


if __name__ == "__main__":
    main()