"""
Test PDF ingestion on any file passed as a command-line argument.
Usage:
    python scripts/test_pdf_ingestion.py                       # defaults to sample_contract.pdf
    python scripts/test_pdf_ingestion.py path/to/contract.pdf  # any PDF
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest


def main():
    # --- Determine which PDF to test ---
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1]).resolve()
    else:
        pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_contract.pdf"

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        print("   Did you download a real PDF into data/raw/?")
        return

    print("=" * 60)
    print(f"Ingesting: {pdf_path.name}")
    print(f"Full path: {pdf_path}")
    print("=" * 60)

    try:
        doc = ingest(pdf_path)
    except Exception as e:
        print(f"❌ Ingestion failed: {type(e).__name__}: {e}")
        return

    print(f"Format:   {doc.format}")
    print(f"Pages:    {len(doc.pages)}")
    print(f"Chars:    {len(doc.full_text)}")
    print(f"Metadata: {doc.metadata.get('ingestion', {})}")
    print()

    # Show first 500 chars
    preview = doc.full_text[:500]
    print("First 500 chars of cleaned text:")
    print("-" * 60)
    print(preview)
    print("-" * 60)

    # Show last 200 chars to see if truncation/trailing content is handled
    if len(doc.full_text) > 700:
        print()
        print("Last 200 chars (check nothing got cut off):")
        print("-" * 60)
        print(doc.full_text[-200:])
        print("-" * 60)

    # Basic sanity checks
    print()
    print("Sanity checks:")
    print(f"  Non-empty text:       {bool(doc.full_text.strip())}")
    print(f"  Reasonable length:    {len(doc.full_text) > 200}")
    print(f"  Looks like a contract: {'agreement' in doc.full_text.lower() or 'contract' in doc.full_text.lower()}")


if __name__ == "__main__":
    main()