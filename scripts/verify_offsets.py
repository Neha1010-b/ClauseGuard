"""
Verify that clause char_start/char_end offsets correctly map back
into the cleaned full_text. This is critical for Phase 8 highlighting.
Run: python scripts/verify_offsets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation import segment


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"

    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 70)
    print("Offset Verification Test")
    print("=" * 70)

    doc = ingest(pdf_path)
    clauses = segment(doc.full_text)

    print(f"\nTotal clauses: {len(clauses)}")
    print()

    # Test every single clause — not just one
    all_ok = True
    for c in clauses:
        sliced = doc.full_text[c.char_start:c.char_end]
        if sliced != c.text:
            all_ok = False
            print(f"  ❌ MISMATCH at clause #{c.id}")
            print(f"     stored: {repr(c.text[:60])}")
            print(f"     sliced: {repr(sliced[:60])}")

    if all_ok:
        print(f"  ✅ All {len(clauses)} clauses have correct offsets")
    else:
        print("  ❌ Some clauses have incorrect offsets — see above")

    # Show one example in detail
    print()
    print("Detailed example — clause #4:")
    c = clauses[4]
    print(f"  number:      {c.number}")
    print(f"  role:        {c.structural_role}")
    print(f"  char range:  [{c.char_start}, {c.char_end}]")
    print(f"  length:      {c.length} (chars)")
    print(f"  text preview:")
    print(f"  {repr(c.text[:150])}")

    # Show hierarchy check
    print()
    print("Hierarchy check — first 10 clauses:")
    for c in clauses[:10]:
        parent = f"parent={c.parent_id}" if c.parent_id is not None else "parent=None"
        children = f"children={c.child_ids}" if c.child_ids else "children=[]"
        print(f"  #{c.id:02d} lvl={c.level} {parent:12} {children}")


if __name__ == "__main__":
    main()