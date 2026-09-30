"""
Phase 5.4 — Test the risk scorer on the real contract.
Shows per-clause risk scores, levels, and categories.

Run: python scripts/test_risk_scorer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation import segment
from src.nlp import ClauseClassifier
from src.risk import ClauseComparator, RiskScorer


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 110)
    print("Risk Scoring — full pipeline output")
    print("=" * 110)
    print()

    doc = ingest(pdf_path)
    clauses = segment(doc.full_text)
    clf = ClauseClassifier()
    classifications = clf.classify_clauses(clauses, top_k=3)
    cmp = ClauseComparator()
    comparisons = cmp.compare_batch(clauses, predictions=classifications)
    scorer = RiskScorer()
    risks = scorer.score_batch(clauses, classifications, comparisons)

    # ---- Sort by risk descending ----
    rows = []
    for c, cls, cm, rk in zip(clauses, classifications, comparisons, risks):
        rows.append({
            "id": c.id,
            "number": c.number or c.structural_role[:4],
            "text": c.text,
            "label": cls["label"],
            "confidence": cls["confidence"],
            "deviation": cm["deviation_score"],
            "mismatch": cm.get("agrees_with_classifier"),
            "risk_score": rk["risk_score"],
            "risk_level": rk["risk_level"],
            "risk_categories": rk["risk_categories"],
            "explanation_hint": rk["explanation_hint"],
            "skipped": cls.get("skipped", False),
        })

    rows.sort(key=lambda r: (r["skipped"], -r["risk_score"]))

    # ---- Detailed output ----
    print("=" * 110)
    print("RISK RANKING (highest risk first)")
    print("=" * 110)
    print()

    level_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for r in rows:
        if r["skipped"]:
            continue
        icon = level_icons[r["risk_level"]]
        cats = ",".join(r["risk_categories"]) if r["risk_categories"] else "—"

        text_preview = r["text"][:65].replace("\n", " ").strip()
        if len(r["text"]) > 65:
            text_preview += "..."

        print(f"  {icon}  #{r['id']:>2}  [{r['number']:>6}]  "
              f"risk={r['risk_score']:.3f}  level={r['risk_level']:<6}  "
              f"cats=[{cats}]")
        print(f"        label={r['label']:<22} "
              f"conf={r['confidence']:.2f}  "
              f"dev={r['deviation']:.3f}  "
              f"mismatch={'Y' if r['mismatch'] is False else 'N'}")
        print(f"        {text_preview}")
        print(f"        → {r['explanation_hint']}")
        print()

    # ---- Summary ----
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    substantive = [r for r in rows if not r["skipped"]]
    by_level = {"high": [], "medium": [], "low": []}
    for r in substantive:
        by_level[r["risk_level"]].append(r)

    print(f"\nSubstantive clauses: {len(substantive)}")
    for level in ("high", "medium", "low"):
        print(f"  {level_icons[level]}  {level:<6}: {len(by_level[level]):>2}")

    print(f"\nRisk score stats:")
    scores = [r["risk_score"] for r in substantive]
    if scores:
        print(f"  Min:    {min(scores):.3f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.3f}")
        print(f"  Mean:   {sum(scores)/len(scores):.3f}")
        print(f"  Max:    {max(scores):.3f}")

    print(f"\nCategory counts:")
    from collections import Counter
    all_cats = []
    for r in substantive:
        all_cats.extend(r["risk_categories"])
    for cat, cnt in Counter(all_cats).most_common():
        print(f"  {cat:<12}: {cnt}")

    # ---- Top 5 most risky ----
    print(f"\nTop 5 highest-risk clauses:")
    for r in substantive[:5]:
        print(f"  {level_icons[r['risk_level']]}  #{r['id']}  [{r['number']}]  "
              f"risk={r['risk_score']:.3f}  {r['label']}")
        print(f"     {r['text'][:100].strip()}...")


if __name__ == "__main__":
    main()