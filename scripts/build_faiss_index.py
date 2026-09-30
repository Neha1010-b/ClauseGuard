"""
Phase 4.2 — Build FAISS index from the reference bank.

Embeds the 500 reference exemplars with MiniLM, builds a FAISS index,
and saves both the index and metadata for later retrieval.

Run: python scripts/build_faiss_index.py
Runtime: <10 seconds (only 500 embeddings).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.config import get_config, PROJECT_ROOT


def main():
    cfg = get_config()
    ref_dir = PROJECT_ROOT / cfg["paths"]["data_reference"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]

    bank_path = ref_dir / "reference_bank.json"
    index_path = models_dir / cfg["reference_bank"]["index_path"].split("/")[-1]
    metadata_path = models_dir / cfg["reference_bank"]["metadata_path"].split("/")[-1]

    print("=" * 80)
    print("Phase 4.2 — Building FAISS index from reference bank")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load reference bank
    # ---------------------------------------------------------
    if not bank_path.exists():
        raise FileNotFoundError(
            f"Reference bank not found at {bank_path}. "
            "Run scripts/build_reference_bank.py first."
        )

    with open(bank_path, "r", encoding="utf-8") as f:
        bank_data = json.load(f)

    exemplars = bank_data["exemplars"]
    print(f"\n[1/4] Loaded {len(exemplars)} exemplars "
          f"across {bank_data['num_labels']} labels")

    # ---------------------------------------------------------
    # 2. Embed exemplars
    # ---------------------------------------------------------
    print(f"\n[2/4] Embedding exemplars with {cfg['embeddings']['model_name']}...")
    model = SentenceTransformer(cfg["embeddings"]["model_name"])

    texts = [e["text"] for e in exemplars]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalized -> inner product == cosine
    ).astype(np.float32)

    dim = embeddings.shape[1]
    print(f"      Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------------
    # 3. Build FAISS index
    # ---------------------------------------------------------
    print(f"\n[3/4] Building FAISS index...")
    # We use IndexFlatIP: inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"      Index contains {index.ntotal} vectors (dim={dim})")

    # ---------------------------------------------------------
    # 4. Save index + metadata
    # ---------------------------------------------------------
    models_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata = {
        "embedding_model": cfg["embeddings"]["model_name"],
        "embedding_dim": dim,
        "num_vectors": int(index.ntotal),
        "metric": "cosine (via IndexFlatIP on normalized vectors)",
        "exemplars": [
            {
                "idx": i,
                "label": e["label"],
                "label_id": e["label_id"],
                "text": e["text"],
                "similarity_to_centroid": e["similarity_to_centroid"],
            }
            for i, e in enumerate(exemplars)
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[4/4] Saved:")
    print(f"      Index:    {index_path}  ({index_path.stat().st_size / 1024:.1f} KB)")
    print(f"      Metadata: {metadata_path}  ({metadata_path.stat().st_size / 1024:.1f} KB)")

    # ---------------------------------------------------------
    # Quick sanity check — self-similarity should be ~1.0
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Sanity check — retrieve nearest exemplar for a sample query")
    print("=" * 80)

    sample_query = (
        "The Receiving Party shall hold all Confidential Information in strict "
        "confidence and shall not disclose it to any third party."
    )
    q_emb = model.encode([sample_query], convert_to_numpy=True,
                         normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(q_emb, k=3)

    print(f"\nQuery: {sample_query[:100]}...")
    print(f"\nTop-3 nearest exemplars:")
    for rank, (sim, idx) in enumerate(zip(distances[0], indices[0]), 1):
        label = metadata["exemplars"][idx]["label"]
        text = metadata["exemplars"][idx]["text"][:100]
        print(f"\n  #{rank}  similarity={sim:.4f}  label={label}")
        print(f"      {text}...")

    print("\n" + "=" * 80)
    print("✅ FAISS index built. Next: Phase 4.3 (ClauseComparator).")
    print("=" * 80)


if __name__ == "__main__":
    main()