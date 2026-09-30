"""
Phase 4.4 — Test the comparator on the real contract.
Shows which clauses deviate most from the reference bank.

Run: python scripts/test_comparator.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation import segment
from src.nlp import ClauseClassifier
from src.risk import ClauseComparator


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 100)
    print("Full pipeline + comparison: PDF -> clauses -> labels -> deviation scores")
    print("=" * 100)
    print()

    doc = ingest(pdf_path)
    clauses = segment(doc.full_text)
    clf = ClauseClassifier()
    preds = clf.classify_clauses(clauses, top_k=3)
    cmp = ClauseComparator()
    comparisons = cmp.compare_batch(clauses, predictions=preds)

    # ---- Detailed table ----
    print("=" * 100)
    print("CLAUSE ANALYSIS (sorted by deviation — most deviant first)")
    print("=" * 100)
    print()

    # Build combined rows
    rows = []
    for c, p, cp in zip(clauses, preds, comparisons):
        rows.append({
            "id": c.id,
            "number": c.number or c.structural_role[:4],
            "text": c.text,
            "label": p["label"],
            "confidence": p["confidence"],
            "skipped": p.get("skipped", False),
            "max_sim": cp["max_similarity"],
            "deviation": cp["deviation_score"],
            "top_match_label": cp["nearest_exemplars"][0]["label"] if cp["nearest_exemplars"] else "—",
            "agrees": cp["agrees_with_classifier"],
        })

    # Sort by deviation descending, but keep structural clauses at bottom
    rows.sort(key=lambda r: (r["skipped"], -r["deviation"]))

    for r in rows:
        text_preview = r["text"][:60].replace("\n", " ").strip()
        if len(r["text"]) > 60:
            text_preview += "..."

        if r["skipped"]:
            tag = "SKIP"
        elif r["agrees"] is True:
            tag = "✓   "
        elif r["agrees"] is False:
            tag = "MISMATCH"
        else:
            tag = "?    "

        print(f"  #{r['id']:>2}  [{r['number']:>6}]  {tag:<9}  "
              f"{r['label']:<22} conf={r['confidence']:.2f}  "
              f"max_sim={r['max_sim']:.3f}  dev={r['deviation']:.3f}")
        print(f"         {text_preview}")
        print()

    # ---- Summary ----
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    substantive = [r for r in rows if not r["skipped"]]
    devs = [r["deviation"] for r in substantive]

    if devs:
        print(f"\nDeviation stats (substantive clauses, n={len(substantive)}):")
        print(f"  Min:     {min(devs):.3f}")
        print(f"  Median:  {sorted(devs)[len(devs)//2]:.3f}")
        print(f"  Mean:    {sum(devs)/len(devs):.3f}")
        print(f"  Max:     {max(devs):.3f}")

        # Histogram-style buckets
        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for d in devs:
            if d < 0.2: buckets["0.0-0.2"] += 1
            elif d < 0.4: buckets["0.2-0.4"] += 1
            elif d < 0.6: buckets["0.4-0.6"] += 1
            elif d < 0.8: buckets["0.6-0.8"] += 1
            else: buckets["0.8-1.0"] += 1
        print(f"\n  Deviation distribution:")
        for k, v in buckets.items():
            bar = "█" * v
            print(f"    {k}:  {v:>2}  {bar}")

    mismatches = [r for r in substantive if r["agrees"] is False]
    print(f"\n  Classifier/exemplar label mismatches: {len(mismatches)}")
    for r in mismatches[:5]:
        print(f"    #{r['id']}: classifier={r['label']}, nearest_exemplar={r['top_match_label']}")


if __name__ == "__main__":
    main()