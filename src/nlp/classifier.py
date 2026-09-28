"""
Clause classifier — Phase 3.6
Wraps the fine-tuned DistilBERT model as a clean inference API.

Design:
- Model loaded once per process (cached)
- Config-driven (path, batch size, max length from config.yaml)
- Returns rich predictions (label + confidence + top-k)
- Fails loudly on missing model directory

Public API:
    clf = ClauseClassifier()
    clf.predict(text) -> dict
    clf.predict_batch(texts) -> List[dict]
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils.config import get_config, PROJECT_ROOT


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

    # Load label names
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

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"[classifier] Model loaded on {device} — {len(label_names)} labels")
    return model, tokenizer, label_names, device


class ClauseClassifier:
    """
    Clause type classifier.

    Usage:
        clf = ClauseClassifier()
        result = clf.predict("The Consultant shall not disclose...")
        results = clf.predict_batch([...])
    """

    def __init__(self):
        cfg = get_config()["classification"]
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.model, self.tokenizer, self.label_names, self.device = _load_model_and_tokenizer()

    # ---------------- Single prediction ----------------
    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify a single clause.
        Returns dict with: label, confidence, top_k (list of (label, prob)).
        """
        if not text or not text.strip():
            return {"label": None, "confidence": 0.0, "top_k": []}

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1)[0]
        top_k = min(top_k, len(self.label_names))

        top_probs, top_indices = torch.topk(probs, k=top_k)

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
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Classify many clauses efficiently. Uses mini-batching for speed.
        """
        if not texts:
            return []

        results: List[Dict[str, Any]] = []
        n = len(texts)
        bs = self.batch_size

        iterator = range(0, n, bs)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="[classifier] predicting", unit="batch")
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i : i + bs]
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