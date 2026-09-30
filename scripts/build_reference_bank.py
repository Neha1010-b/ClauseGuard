"""
Phase 4.1 — Build the reference clause bank.

For each of the 100 LEDGAR clause types, select the K most representative
clauses (medoids) from the training set. These become the "standard" for
that clause type — later, we compare incoming clauses against these.

Run: python scripts/build_reference_bank.py
Runtime: ~15-30 min on CPU (first run), <1 min after (cached).
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config import get_config, PROJECT_ROOT


# How many exemplars to pick per label
K_PER_LABEL = 5

# Cosine similarity floor between selected exemplars — avoids picking near-duplicates
DIVERSITY_THRESHOLD = 0.85

# Embedding batch size
BATCH_SIZE = 64


def main():
    cfg = get_config()
    ref_dir = PROJECT_ROOT / cfg["paths"]["data_reference"]
    ref_dir.mkdir(parents=True, exist_ok=True)

    embeddings_cache = ref_dir / "ledgar_train_embeddings.npy"
    labels_cache = ref_dir / "ledgar_train_labels.npy"
    bank_path = ref_dir / "reference_bank.json"

    print("=" * 80)
    print("Phase 4.1 — Building reference bank from LEDGAR")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load LEDGAR
    # ---------------------------------------------------------
    print("\n[1/5] Loading LEDGAR-100...")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    train = ds["train"]
    label_names = train.features["label"].names
    print(f"      {len(train):,} training clauses, {len(label_names)} labels")

    train_texts = train["text"]
    train_labels = np.array(train["label"])

    # ---------------------------------------------------------
    # 2. Embed (with cache)
    # ---------------------------------------------------------
    if embeddings_cache.exists() and labels_cache.exists():
        print(f"\n[2/5] Loading cached embeddings from {embeddings_cache.name}")
        embeddings = np.load(embeddings_cache)
        cached_labels = np.load(labels_cache)
        if len(cached_labels) != len(train_labels):
            print("      Cache size mismatch — re-computing...")
            embeddings = None
        else:
            print(f"      Loaded {embeddings.shape[0]:,} embeddings "
                  f"(dim={embeddings.shape[1]})")
    else:
        embeddings = None

    if embeddings is None:
        print("\n[2/5] Computing embeddings (this takes ~15-30 min on CPU)...")
        model = SentenceTransformer(cfg["embeddings"]["model_name"])
        embeddings = []
        start = time.time()
        for i in tqdm(range(0, len(train_texts), BATCH_SIZE),
                      desc="      embedding"):
            batch = train_texts[i:i + BATCH_SIZE]
            vecs = model.encode(batch, convert_to_numpy=True,
                                show_progress_bar=False, normalize_embeddings=True)
            embeddings.append(vecs)
        embeddings = np.vstack(embeddings).astype(np.float32)
        elapsed = time.time() - start
        print(f"      Done in {elapsed:.0f}s — shape {embeddings.shape}")
        np.save(embeddings_cache, embeddings)
        np.save(labels_cache, train_labels)
        print(f"      Cached to {embeddings_cache.name}")

    # ---------------------------------------------------------
    # 3. Group by label
    # ---------------------------------------------------------
    print("\n[3/5] Grouping embeddings by label...")
    by_label = defaultdict(list)
    for idx, lbl in enumerate(train_labels):
        by_label[int(lbl)].append(idx)
    print(f"      {len(by_label)} distinct labels found")

    # ---------------------------------------------------------
    # 4. Select medoids per label (with diversity filter)
    # ---------------------------------------------------------
    print(f"\n[4/5] Selecting up to {K_PER_LABEL} exemplars per label "
          f"(diversity threshold={DIVERSITY_THRESHOLD})...")
    reference_bank = []

    for lbl_id, indices in tqdm(sorted(by_label.items()), desc="      labels"):
        embs = embeddings[indices]
        # Centroid of this label
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12

        # Cosine similarity to centroid
        sims = embs @ centroid  # vectors are normalized

        # Sort by similarity descending
        order = np.argsort(-sims)

        selected = []
        selected_embs = []

        for oi in order:
            idx = indices[oi]
            emb = embeddings[idx]

            # Diversity check
            if selected_embs:
                sim_to_selected = np.max(
                    np.array(selected_embs) @ emb
                )
                if sim_to_selected > DIVERSITY_THRESHOLD:
                    continue

            selected.append({
                "label_id": lbl_id,
                "label": label_names[lbl_id],
                "text": train_texts[idx],
                "similarity_to_centroid": float(sims[oi]),
                "original_index": int(idx),
            })
            selected_embs.append(emb)

            if len(selected) >= K_PER_LABEL:
                break

        reference_bank.extend(selected)

    print(f"\n      Selected {len(reference_bank)} total exemplars")

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump({
            "k_per_label": K_PER_LABEL,
            "diversity_threshold": DIVERSITY_THRESHOLD,
            "embedding_model": cfg["embeddings"]["model_name"],
            "num_labels": len(label_names),
            "label_names": label_names,
            "num_exemplars": len(reference_bank),
            "exemplars": reference_bank,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[5/5] Saved to {bank_path}")
    print(f"      Size: {bank_path.stat().st_size / 1024:.1f} KB")

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAMPLE — 2 exemplars for a few labels")
    print("=" * 80)
    from itertools import groupby
    preview_labels = {"Confidentiality", "Terminations", "Governing Laws", "Notices"}
    for item in reference_bank:
        if item["label"] in preview_labels:
            print(f"\n[{item['label']}] (sim_to_centroid={item['similarity_to_centroid']:.3f})")
            print(f"  {item['text'][:180]}...")

    print("\n" + "=" * 80)
    print("✅ Reference bank built. Next: Phase 4.2 (embed + index).")
    print("=" * 80)


if __name__ == "__main__":
    main()