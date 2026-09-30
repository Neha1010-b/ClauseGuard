"""
Phase 5.6 — Sanity check: does the risk scorer flag known high-risk clauses?

We feed it clauses that any contract lawyer would flag as risky,
plus some clearly standard clauses, and verify the scorer separates them.

Run: python scripts/test_high_risk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp import ClauseClassifier
from src.risk import ClauseComparator, RiskScorer
from src.segmentation.clause import Clause, ROLE_CLAUSE


HIGH_RISK = [
    ("The Company may terminate this Agreement at any time, with or without cause, "
     "in its sole discretion, without prior notice and without any liability.", "Terminations"),
    ("The Consultant shall indemnify, defend, and hold harmless the Company from any "
     "and all claims, damages, losses, liabilities, costs, and expenses of any kind "
     "whatsoever, whether direct or indirect, arising out of or related to this Agreement, "
     "without limitation.", "Indemnifications"),
    ("The Consultant hereby irrevocably assigns to the Company all right, title, and "
     "interest in any inventions, discoveries, improvements, or works of authorship "
     "conceived or developed during the term of this Agreement, whether or not during "
     "working hours, without additional compensation.", "Intellectual Property"),
    ("During the term of this Agreement and for a period of five (5) years thereafter, "
     "the Consultant shall not, anywhere in India, directly or indirectly, engage in "
     "any business that competes with the Company.", "Employment"),
    ("This Agreement shall automatically renew for successive one (1) year terms unless "
     "either party provides written notice of non-renewal at least one hundred eighty "
     "(180) days prior to the end of the then-current term.", "Terms"),
]

STANDARD = [
    ("Either party may terminate this Agreement upon thirty (30) days' prior written notice "
     "to the other party.", "Terminations"),
    ("This Agreement shall be governed by and construed in accordance with the laws of "
     "the State of California.", "Governing Laws"),
    ("All notices under this Agreement shall be in writing and delivered by hand, "
     "courier, or certified mail.", "Notices"),
    ("Each party shall keep confidential all non-public information received from the "
     "other party and shall not disclose it to any third party.", "Confidentiality"),
    ("If any provision of this Agreement is held invalid or unenforceable, the remaining "
     "provisions shall remain in full force and effect.", "Severability"),
]


def build_clause(i: int, text: str) -> Clause:
    return Clause(
        id=i, text=text, char_start=0, char_end=len(text),
        level=1, structural_role=ROLE_CLAUSE,
    )


def run_batch(clf, cmp, scorer, items, label):
    print(f"\n{'=' * 90}")
    print(f"{label}")
    print("=" * 90)
    clauses = [build_clause(i, text) for i, (text, _) in enumerate(items)]
    classifications = clf.classify_clauses(clauses, top_k=3, show_progress=False)
    comparisons = cmp.compare_batch(clauses, predictions=classifications, show_progress=False)
    risks = scorer.score_batch(clauses, classifications, comparisons)

    for c, cls, cm, rk in zip(clauses, classifications, comparisons, risks):
        print(f"\n  risk={rk['risk_score']:.3f}  level={rk['risk_level']:<6}  "
              f"cats={rk['risk_categories']}")
        print(f"  label={cls['label']} (conf={cls['confidence']:.2f})  "
              f"dev={cm['deviation_score']:.3f}")
        print(f"  {c.text[:100]}")
        print(f"  → {rk['explanation_hint']}")


def main():
    clf = ClauseClassifier()
    cmp = ClauseComparator()
    scorer = RiskScorer()

    run_batch(clf, cmp, scorer, HIGH_RISK, "HIGH-RISK CLAUSES (should be flagged)")
    run_batch(clf, cmp, scorer, STANDARD, "STANDARD CLAUSES (should be low risk)")


if __name__ == "__main__":
    main()