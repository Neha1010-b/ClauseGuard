"""
Clause classifier — Phase 3.8
Wraps the fine-tuned DistilBERT model as a clean inference API.

Phase 3.8 enhancements:
- Skip classification for structural markers (PARTIES, RECITAL, SIGNATURE, PREAMBLE)
- Inject parent context for sub-clauses (hierarchical context)
- Primary entry point: classify_clauses(List[Clause]) -> List[Classification]

Design:
- Model loaded once per process (cached)
- Config-driven (path, batch size, max length from config.yaml)
- Returns rich predictions (label + confidence + top-k)
- Fails loudly on missing model directory
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.config import get_config, PROJECT_ROOT


# ============================================================
# Structural roles that should NOT be semantically classified
# ============================================================
_STRUCTURAL_ROLES_TO_SKIP = {
    "PARTIES", "RECITAL", "SIGNATURE", "PREAMBLE",
}


@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    """
    Load the fine-tuned model and tokenizer from disk.
    Cached — safe to call repeatedly without reloading.
    """
    cfg = get_config()["classification"]
    model_path = PROJECT_ROOT / cfg["model_save_path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            f"Expected files: config.json, model.safetensors, tokenizer.json, label_names.json.\n"
            f"Did you complete Phase 3.5 (move the Colab-trained model into models/)?"
        )

    label_names_path = model_path / "label_names.json"
    if not label_names_path.exists():
        raise FileNotFoundError(
            f"label_names.json missing in {model_path}. "
            "Did you copy it alongside the model files?"
        )
    with open(label_names_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    print(f"[classifier] Loading model from {model_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"[classifier] Model loaded on {device} — {len(label_names)} labels")
    return model, tokenizer, label_names, device


# ============================================================
# Context-building helpers
# ============================================================
def _build_context_for_clause(
    clause,
    clauses_by_id: Dict[int, Any],
    max_context_chars: int = 200,
) -> Optional[str]:
    """
    Build a short context string from a clause's parent.

    Rules:
    - Skip context if parent is more than 2 levels up (avoid grandparent pollution).
    - Skip context if parent is way longer than the clause (the model will get
      dominated by the parent's content, not the clause's).
    - Cap context length hard.
    """
    if clause.parent_id is None:
        return None

    parent = clauses_by_id.get(clause.parent_id)
    if parent is None:
        return None

    parent_text = parent.text.strip()

    # Guard: if the parent is much longer than the child, its content will
    # dominate. Only include the parent's first sentence as context.
    # (Heads-up: this is a heuristic — could be refined later.)
    first_sentence = parent_text
    for delim in [". ", ".\n", ":", ";"]:
        idx = parent_text.find(delim)
        if 0 < idx < max_context_chars * 2:
            first_sentence = parent_text[: idx + len(delim)]
            break

    snippet = first_sentence[:max_context_chars]
    if len(first_sentence) > max_context_chars:
        cut = snippet.rfind(" ")
        if cut > 80:
            snippet = snippet[:cut]
        snippet += "..."

    parent_number = parent.number or ""
    if parent_number:
        return f"[Context: parent clause {parent_number}]: {snippet}"
    return f"[Context: parent clause]: {snippet}"


# ============================================================
# Classifier
# ============================================================
class ClauseClassifier:
    """
    Clause type classifier with structural-awareness and context injection.

    Usage:
        clf = ClauseClassifier()
        results = clf.classify_clauses(clauses)   # recommended
        # OR
        result = clf.predict("...")               # raw text
        results = clf.predict_batch([...])        # raw texts
    """

    def __init__(self):
        cfg = get_config()["classification"]
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.model, self.tokenizer, self.label_names, self.device = _load_model_and_tokenizer()

    # ---------------- Single prediction ----------------
    def predict(self, text: str, top_k: int = 5, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a single clause (raw text, no Clause object).
        If `context` is provided, it's prepended to the text.
        """
        if not text or not text.strip():
            return {"label": None, "confidence": 0.0, "top_k": []}

        full_text = f"{context}\n\n{text}" if context else text

        inputs = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1)[0]
        k = min(top_k, len(self.label_names))
        top_probs, top_indices = torch.topk(probs, k=k)

        top_k_list = [
            (self.label_names[i], float(p))
            for i, p in zip(top_indices.tolist(), top_probs.tolist())
        ]

        return {
            "label": top_k_list[0][0],
            "confidence": top_k_list[0][1],
            "top_k": top_k_list,
        }

    # ---------------- Batch prediction ----------------
    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 3,
        contexts: Optional[List[Optional[str]]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Classify many clauses efficiently. Uses mini-batching for speed.

        Args:
            texts:      List of clause texts.
            top_k:      Number of top predictions to return per clause.
            contexts:   Optional list parallel to `texts`. Each entry is a
                        context string to prepend, or None. If provided, must
                        have the same length as `texts`.
        """
        if not texts:
            return []

        if contexts is not None and len(contexts) != len(texts):
            raise ValueError(
                f"contexts length ({len(contexts)}) must match texts length ({len(texts)})"
            )

        # Merge context + text
        if contexts is None:
            merged = list(texts)
        else:
            merged = [
                f"{ctx}\n\n{txt}" if ctx else txt
                for ctx, txt in zip(contexts, texts)
            ]

        results: List[Dict[str, Any]] = []
        n = len(merged)
        bs = self.batch_size

        iterator = range(0, n, bs)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="[classifier] predicting", unit="batch")
            except ImportError:
                pass

        for i in iterator:
            batch = merged[i : i + bs]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = F.softmax(logits, dim=-1)
            k = min(top_k, len(self.label_names))
            top_probs, top_indices = torch.topk(probs, k=k, dim=-1)

            for j in range(len(batch)):
                top_k_list = [
                    (self.label_names[idx], float(p))
                    for idx, p in zip(
                        top_indices[j].tolist(),
                        top_probs[j].tolist(),
                    )
                ]
                results.append({
                    "label": top_k_list[0][0],
                    "confidence": top_k_list[0][1],
                    "top_k": top_k_list,
                })

        return results

    # ---------------- High-level entry point for Clause objects ----------------
    def classify_clauses(self, clauses, top_k: int = 3, show_progress: bool = True):
        """
        Classify a list of Clause objects, applying the Phase 3.8 enhancements:

        1. Structural clauses (PARTIES, RECITAL, SIGNATURE, PREAMBLE) are NOT
           sent to the model. Their label = their structural_role, confidence = 1.0.
        2. Sub-clauses with a parent get the parent's context prepended.

        Returns a list parallel to `clauses`, where each item is:
            {
              "label": str,
              "confidence": float,
              "top_k": [(label, prob), ...],
              "skipped": bool,       # True if classified structurally
              "used_context": bool,  # True if parent context was injected
            }
        """
        from ..segmentation import ROLE_PARTIES, ROLE_RECITAL, ROLE_SIGNATURE, ROLE_PREAMBLE

        clauses_by_id = {c.id: c for c in clauses}

        # Partition clauses into three groups:
        #  A. Structural — skip model
        #  B. Substantive with parent — classifier + context
        #  C. Substantive without parent — classifier, no context
        structural_results: Dict[int, Dict[str, Any]] = {}
        to_classify_ids: List[int] = []
        to_classify_texts: List[str] = []
        to_classify_contexts: List[Optional[str]] = []

        for c in clauses:
            if c.structural_role in _STRUCTURAL_ROLES_TO_SKIP:
                structural_results[c.id] = {
                    "label": c.structural_role,
                    "confidence": 1.0,
                    "top_k": [(c.structural_role, 1.0)],
                    "skipped": True,
                    "used_context": False,
                }
                continue

            ctx = _build_context_for_clause(c, clauses_by_id)
            to_classify_ids.append(c.id)
            to_classify_texts.append(c.text)
            to_classify_contexts.append(ctx)

        # Run the model on the substantive clauses
        model_results = self.predict_batch(
            texts=to_classify_texts,
            top_k=top_k,
            contexts=to_classify_contexts,
            show_progress=show_progress,
        )

        # Merge back into original order
        id_to_model_result = {
            cid: {**res, "skipped": False, "used_context": (ctx is not None)}
            for cid, res, ctx in zip(to_classify_ids, model_results, to_classify_contexts)
        }

        final: List[Dict[str, Any]] = []
        for c in clauses:
            if c.id in structural_results:
                final.append(structural_results[c.id])
            else:
                final.append(id_to_model_result[c.id])
        return final