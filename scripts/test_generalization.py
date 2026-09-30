"""
Phase 3.9 — generalization test.
Run the classifier on clauses it has NEVER seen, from contract genres
different from both LEDGAR and the sample contract.

If it performs well here, the model generalizes — critical for the
research claim that clause-level classification transfers across domains.

Run: python scripts/test_generalization.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp import ClauseClassifier


# Unseen test cases, curated across contract types.
# Each: (clause_text, expected_label_approx)
TEST_CASES = [
    (
        "The Employee shall not, during the term of this Agreement or for "
        "a period of two (2) years thereafter, directly or indirectly engage "
        "in any business competitive with the Company.",
        "Non-Compete",
    ),
    (
        "All payments due under this Agreement shall be made in United States "
        "Dollars by wire transfer to the account designated by the Provider.",
        "Payments",
    ),
    (
        "This Agreement may be terminated by either party upon thirty (30) "
        "days' written notice to the other party.",
        "Terminations",
    ),
    (
        "The Vendor shall indemnify, defend, and hold harmless the Client from "
        "any and all claims arising out of the Vendor's performance under this "
        "Agreement.",
        "Indemnifications",
    ),
    (
        "This Agreement shall be governed by the laws of the State of New York, "
        "without regard to its conflict of laws principles.",
        "Governing Laws",
    ),
    (
        "Neither party shall be liable for any indirect, incidental, special, "
        "or consequential damages arising out of this Agreement.",
        "Limitation of Liability",
    ),
    (
        "The Company shall maintain in full force and effect, during the term "
        "of this Agreement, comprehensive general liability insurance with "
        "limits of not less than $1,000,000 per occurrence.",
        "Insurances",
    ),
    (
        "All notices, requests, consents, claims, demands, waivers, and other "
        "communications under this Agreement shall be in writing and shall be "
        "deemed to have been given when delivered personally or by email.",
        "Notices",
    ),
    (
        "Each party shall keep confidential and shall not disclose to any third "
        "party any non-public information regarding the other party's business, "
        "customers, or technology.",
        "Confidentiality",
    ),
    (
        "No amendment to this Agreement shall be effective unless it is in "
        "writing and signed by both parties.",
        "Amendments",
    ),
]


def main():
    print("=" * 85)
    print("Generalization Test — classifying completely unseen clauses")
    print("=" * 85)
    print()

    clf = ClauseClassifier()
    print()

    correct = 0
    partial = 0
    wrong = 0

    for text, expected in TEST_CASES:
        result = clf.predict(text, top_k=3)
        top_label = result["label"]
        top_conf = result["confidence"]

        # Consider correct if expected is the top prediction or in top-3
        top3_labels = [lbl for lbl, _ in result["top_k"]]

        if top_label == expected:
            mark = "✅"
            correct += 1
        elif expected in top3_labels:
            mark = "🟡"
            partial += 1
        else:
            mark = "❌"
            wrong += 1

        print(f"{mark} Expected: {expected:<25}")
        print(f"   Predicted: {top_label:<25}  ({top_conf:.1%})")
        if top_label != expected:
            alts = ", ".join(f"{l}={p:.1%}" for l, p in result["top_k"][1:3])
            print(f"   Alternatives: {alts}")
        print()

    total = len(TEST_CASES)
    print("=" * 85)
    print(f"RESULTS: {correct}/{total} exact, {partial} in top-3, {wrong} wrong")
    print(f"  Top-1 accuracy: {correct/total:.1%}")
    print(f"  Top-3 accuracy: {(correct+partial)/total:.1%}")
    print("=" * 85)


if __name__ == "__main__":
    main()