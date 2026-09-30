"""
Clause comparator — Phase 4.3
Compares an incoming clause against the reference bank via FAISS.

Public API:
    comparator = ClauseComparator()
    result = comparator.compare(clause_text, predicted_label=None)
    results = comparator.compare_batch(clauses)   # List[Clause] -> List[dict]

Design:
- Loads FAISS index + metadata once (cached)
- Uses the SAME MiniLM model used to build the index (critical!)
- L2-normalizes query embedding for exact cosine via inner product
- Returns deviation_score = 1 - max_similarity
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..utils.config import get_config, PROJECT_ROOT


# ============================================================
# Cached loaders
# ============================================================
@lru_cache(maxsize=1)
def _load_faiss_index():
    """Load the FAISS index + metadata once per process."""
    cfg = get_config()
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]

    index_path = models_dir / cfg["reference_bank"]["index_path"].split("/")[-1]
    metadata_path = models_dir / cfg["reference_bank"]["metadata_path"].split("/")[-1]

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_faiss_index.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found at {metadata_path}."
        )

    print(f"[comparator] Loading FAISS index from {index_path.name}...")
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[comparator] Index ready — {index.ntotal} vectors (dim={index.d})")
    return index, metadata


@lru_cache(maxsize=1)
def _load_embedding_model():
    """Load the SAME embedding model that was used to build the index."""
    cfg = get_config()
    model_name = cfg["embeddings"]["model_name"]
    print(f"[comparator] Loading embedding model {model_name}...")
    model = SentenceTransformer(model_name)
    return model


# ============================================================
# Comparator
# ============================================================
class ClauseComparator:
    """
    Compares a clause against the reference bank.

    Usage:
        cmp = ClauseComparator()
        result = cmp.compare("The Receiving Party shall hold...")
        results = cmp.compare_batch(clauses)   # List[Clause] -> List[dict]
    """

    def __init__(self, top_k: int = 5):
        self.index, self.metadata = _load_faiss_index()
        self.embedder = _load_embedding_model()
        self.top_k = top_k

    # ---------------- Single clause ----------------
    def compare(
        self,
        text: str,
        predicted_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare a clause against the reference bank.

        Args:
            text: The clause text.
            predicted_label: The classifier's label (optional). If provided,
                             we also report whether the top match agrees.

        Returns a dict with:
            max_similarity:         highest similarity to any reference
            mean_top3_similarity:   average of top-3 similarities
            deviation_score:        1 - max_similarity (our risk signal)
            nearest_exemplars:      list of {label, text, similarity}
            agrees_with_classifier: True/False/None
        """
        if not text or not text.strip():
            return {
                "max_similarity": 0.0,
                "mean_top3_similarity": 0.0,
                "deviation_score": 1.0,
                "nearest_exemplars": [],
                "agrees_with_classifier": None,
            }

        # Embed the query (normalized -> inner product == cosine)
        q = self.embedder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        k = min(self.top_k, self.index.ntotal)
        similarities, indices = self.index.search(q, k)
        sims = similarities[0].tolist()
        idxs = indices[0].tolist()

        nearest = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            entry = self.metadata["exemplars"][idx]
            nearest.append({
                "label": entry["label"],
                "text": entry["text"],
                "similarity": float(sim),
                "exemplar_idx": int(idx),
            })

        max_sim = nearest[0]["similarity"] if nearest else 0.0
        mean_top3 = float(np.mean([n["similarity"] for n in nearest[:3]])) if nearest else 0.0

        agrees = None
        if predicted_label and nearest:
            agrees = nearest[0]["label"] == predicted_label

        return {
            "max_similarity": float(max_sim),
            "mean_top3_similarity": mean_top3,
            "deviation_score": float(1.0 - max_sim),
            "nearest_exemplars": nearest,
            "agrees_with_classifier": agrees,
        }

    # ---------------- Batch of Clauses ----------------
    def compare_batch(
        self,
        clauses: List,
        predictions: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Compare many Clause objects at once (embeds them in batches).

        Args:
            clauses:      List of Clause objects.
            predictions:  Optional list of classification results (from ClauseClassifier).
                          If provided, each comparison includes agrees_with_classifier.
            show_progress: Show tqdm progress bar.

        Returns a list parallel to `clauses`.
        """
        if not clauses:
            return []

        texts = [c.text for c in clauses]
        labels = None
        if predictions is not None:
            if len(predictions) != len(clauses):
                raise ValueError(
                    f"predictions length ({len(predictions)}) must match "
                    f"clauses length ({len(clauses)})"
                )
            labels = [p.get("label") for p in predictions]

        # Batch embed
        iterator = range(0, len(texts), 64)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="[comparator] embedding", unit="batch")
            except ImportError:
                pass

        all_embeddings = []
        for i in iterator:
            batch = texts[i:i + 64]
            embs = self.embedder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            all_embeddings.append(embs)
        all_embeddings = np.vstack(all_embeddings)

        # Search FAISS in one shot
        k = min(self.top_k, self.index.ntotal)
        similarities, indices = self.index.search(all_embeddings, k)

        results = []
        for i in range(len(clauses)):
            sims = similarities[i].tolist()
            idxs = indices[i].tolist()

            nearest = []
            for sim, idx in zip(sims, idxs):
                if idx < 0:
                    continue
                entry = self.metadata["exemplars"][idx]
                nearest.append({
                    "label": entry["label"],
                    "text": entry["text"],
                    "similarity": float(sim),
                    "exemplar_idx": int(idx),
                })

            max_sim = nearest[0]["similarity"] if nearest else 0.0
            mean_top3 = float(np.mean([n["similarity"] for n in nearest[:3]])) if nearest else 0.0

            agrees = None
            if labels and labels[i] and nearest:
                agrees = nearest[0]["label"] == labels[i]

            results.append({
                "max_similarity": float(max_sim),
                "mean_top3_similarity": mean_top3,
                "deviation_score": float(1.0 - max_sim),
                "nearest_exemplars": nearest,
                "agrees_with_classifier": agrees,
            })

        return results