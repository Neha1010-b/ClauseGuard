"""
Phase 6A.5 — Generate Gemini explanations for risky clauses in the real contract.
Run: python scripts/test_explanations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest
from src.segmentation import segment
from src.nlp import ClauseClassifier
from src.risk import ClauseComparator, RiskScorer
from src.rag import ExplanationEngine


def main():
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    print("=" * 100)
    print("Phase 6A — Full pipeline with Gemini explanations")
    print("=" * 100)
    print()

    # --- Pipeline ---
    doc = ingest(pdf_path)
    clauses = segment(doc.full_text)
    clf = ClauseClassifier()
    classifications = clf.classify_clauses(clauses, top_k=3)
    cmp = ClauseComparator()
    comparisons = cmp.compare_batch(clauses, predictions=classifications)
    scorer = RiskScorer()
    risks = scorer.score_batch(clauses, classifications, comparisons)

    # --- Which clauses qualify for explanation? ---
    engine = ExplanationEngine()
    to_explain = [
        i for i, r in enumerate(risks)
        if engine._should_explain(r["risk_level"])
    ]

    print(f"\nQualifying clauses for LLM explanation: {len(to_explain)}")
    for i in to_explain:
        c = clauses[i]
        r = risks[i]
        print(f"  #{c.id} [{c.number or c.structural_role[:4]}]  "
              f"level={r['risk_level']}  score={r['risk_score']:.3f}")
    print()

    if not to_explain:
        print("No clauses qualify. Nothing to explain.")
        return

    # --- Generate explanations ---
    print("=" * 100)
    print("GENERATING EXPLANATIONS...")
    print("=" * 100)

    explanations = engine.explain_batch(clauses, classifications, comparisons, risks)

    # --- Print results ---
    print()
    print("=" * 100)
    print("EXPLANATIONS")
    print("=" * 100)

    for i in to_explain:
        c = clauses[i]
        cls = classifications[i]
        r = risks[i]
        expl = explanations[i]

        print()
        print("─" * 100)
        print(f"CLAUSE #{c.id}  [{c.number or c.structural_role}]  "
              f"{r['risk_level'].upper()} RISK  (score {r['risk_score']:.3f})")
        print("─" * 100)
        print()
        print(f"Detected type:      {cls['label']}  (conf {cls['confidence']:.2f})")
        if r.get("risk_categories"):
            print(f"Risk categories:    {', '.join(r['risk_categories'])}")
        if r.get("signals", {}).get("risky_language_tags"):
            print(f"Risky language:     {', '.join(r['signals']['risky_language_tags'])}")
        print()
        print("Clause text:")
        # Wrap text at 90 chars
        import textwrap
        for line in textwrap.wrap(c.text.strip(), width=90):
            print(f"  {line}")
        print()

        if expl.get("explanation"):
            print("─" * 100)
            print("AI EXPLANATION:")
            print("─" * 100)
            for line in textwrap.wrap(expl["explanation"], width=90):
                print(f"  {line}")
            print()
            if expl.get("suggested_action"):
                print("SUGGESTED ACTION:")
                for line in textwrap.wrap(expl["suggested_action"], width=90):
                    print(f"  {line}")
                print()
            print(f"(model: {expl['model_used']}, latency: {expl['latency_ms']}ms)")
        else:
            print("⚠️  No explanation generated.")
            print(f"    Reason: {expl.get('skipped_reason')}")


if __name__ == "__main__":
    main()