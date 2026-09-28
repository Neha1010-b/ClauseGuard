"""
Phase 3.1 — inspect the LEDGAR clause-classification dataset.
Downloads once (~50 MB), then prints structure and label distribution.

Run: python scripts/inspect_ledgar.py
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset


def main():
    print("=" * 70)
    print("LEDGAR Dataset Inspection")
    print("=" * 70)

    # LEDGAR is hosted as `lex_glue` (LexGLUE benchmark) with LEDGAR as a subset.
    # We use the "ledgar" config from lex_glue.
    print("\nDownloading / loading LEDGAR (cached after first run)...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")

    print(f"\nSplits available: {list(ds.keys())}")

    # -------- Structure --------
    print("\n" + "-" * 70)
    print("Split sizes:")
    for split_name, split_ds in ds.items():
        print(f"  {split_name:10s}: {len(split_ds):>7,} clauses")

    # -------- Features --------
    print("\n" + "-" * 70)
    print("Features (columns):")
    print(ds["train"].features)

    # -------- Example row --------
    print("\n" + "-" * 70)
    print("Example clause (index 0 from train):")
    row = ds["train"][0]
    print(f"  label:  {row['label']}")
    print(f"  text:   {row['text'][:300]}...")

    # -------- Label distribution --------
    print("\n" + "-" * 70)
    print("Label distribution on TRAIN split:")
    train_labels = ds["train"]["label"]
    counts = Counter(train_labels)

    # Get human-readable label names from the ClassLabel feature
    label_feature = ds["train"].features["label"]
    label_names = label_feature.names
    print(f"  Number of distinct labels: {len(label_names)}")
    print()

    # Sort by frequency
    sorted_labels = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"  {'Rank':<5}{'Count':>8}  {'Name':<30}")
    print(f"  {'-'*5}{'-'*8}  {'-'*30}")
    for rank, (label_id, cnt) in enumerate(sorted_labels, 1):
        name = label_names[label_id] if label_id < len(label_names) else f"<id {label_id}>"
        print(f"  {rank:<5}{cnt:>8,}  {name:<30}")

    # -------- Balance analysis --------
    print("\n" + "-" * 70)
    print("Balance analysis:")
    if sorted_labels:
        top_count = sorted_labels[0][1]
        bottom_count = sorted_labels[-1][1]
        print(f"  Most common class:  {top_count:,}")
        print(f"  Least common class: {bottom_count:,}")
        print(f"  Imbalance ratio:    {top_count / max(bottom_count, 1):.1f}x")
        print("  (Large ratios mean we need class-weighted loss or oversampling.)")

    # -------- Label names (full list) --------
    print("\n" + "-" * 70)
    print(f"All {len(label_names)} label names (alphabetical):")
    for i, name in enumerate(sorted(label_names)):
        print(f"  {i+1:>3}. {name}")


if __name__ == "__main__":
    main()