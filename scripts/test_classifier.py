"""
Phase 3.7 — run the full pipeline: PDF → clauses → classified clauses.
Run: python scripts/test_classifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation import segment
from src.nlp import ClauseClassifier


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 90)
    print("Full Pipeline: PDF -> Clauses -> Classified Clauses")
    print("=" * 90)
    print()

    # Stage 1: ingest
    doc = ingest(pdf_path)
    print(f"[1/3] Ingested: {doc.format} | {len(doc.full_text):,} chars | {len(doc.pages)} pages")
    print()

    # Stage 2: segment
    clauses = segment(doc.full_text)
    print(f"[2/3] Segmented: {len(clauses)} clauses")
    print()

    # Stage 3: classify (with Phase 3.8 enhancements)
    clf = ClauseClassifier()
    predictions = clf.classify_clauses(clauses, top_k=3)
    print(f"[3/3] Classified: {len(predictions)} clauses")
    print()

    # ---- Output table ----
    print("=" * 90)
    print("RESULTS: clause -> predicted type")
    print("=" * 90)
    print()

    for clause, pred in zip(clauses, predictions):
        # First 55 chars of clause text
        preview = clause.text[:55].replace("\n", " ").strip()
        if len(clause.text) > 55:
            preview += "..."

        label = pred["label"] or "—"
        conf = pred["confidence"]

        # Line 1: preview
        print(f"  #{clause.id:>2}  {preview}")

        # Line 2: prediction
        if clause.number:
            num_str = f"[{clause.number}]"
        else:
            num_str = f"[{clause.structural_role[:4]}]"
        print(f"       {num_str:>10}  ->  {label:<28} ({conf:.2%})")

        # Line 3: top-3 alternatives (only if confidence < 0.7)
        if conf < 0.7 and len(pred["top_k"]) > 1:
            alts = ", ".join(
                f"{lbl}={p:.1%}" for lbl, p in pred["top_k"][1:3]
            )
            print(f"                    alternatives: {alts}")
        print()

    # ---- Summary statistics ----
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    from collections import Counter
    label_counts = Counter(p["label"] for p in predictions if p["label"])

    print(f"\nDistinct clause types detected: {len(label_counts)}")
    print(f"\nTop 10 most common:")
    for label, cnt in label_counts.most_common(10):
        print(f"  {cnt:>3}x  {label}")

	# Context & skip stats
    skipped = sum(1 for p in predictions if p.get("skipped"))
    used_ctx = sum(1 for p in predictions if p.get("used_context"))
    print(f"\nContext/skip breakdown:")
    print(f"  Structural (skipped):  {skipped}")
    print(f"  With parent context:   {used_ctx}")
    print(f"  Standalone:            {len(predictions) - skipped - used_ctx}")
    
    # Confidence distribution
    confidences = [p["confidence"] for p in predictions if p["label"]]
    if confidences:
        print(f"\nConfidence stats:")
        print(f"  Min:    {min(confidences):.2%}")
        print(f"  Median: {sorted(confidences)[len(confidences)//2]:.2%}")
        print(f"  Max:    {max(confidences):.2%}")
        print(f"  Mean:   {sum(confidences)/len(confidences):.2%}")

    # High vs low confidence
    high = sum(1 for c in confidences if c >= 0.7)
    low = sum(1 for c in confidences if c < 0.5)
    print(f"\n  High-confidence (>=70%): {high}/{len(confidences)}")
    print(f"  Low-confidence  (<50%):  {low}/{len(confidences)}")


if __name__ == "__main__":
    main()
