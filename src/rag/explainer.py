"""
Explanation Engine — Phase 6A.2
Turns structured risk signals into human-readable explanations via Gemini.

Design:
- Loads Gemini client once (cached)
- Only generates explanations for clauses at or above a risk threshold
- Automatic model fallback: primary -> fallback_1 -> fallback_2 -> ...
- Returns structured output: explanation, suggested_action, model_used
- Never raises on individual clause failure — returns an error placeholder

Public API:
    engine = ExplanationEngine()
    result = engine.explain(clause, classification, comparison, risk)
    results = engine.explain_batch(clauses, classifications, comparisons, risks)
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from ..utils.config import get_config, PROJECT_ROOT


# Load .env at module import so GEMINI_API_KEY is available
_ENV_PATH = PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


# ============================================================
# Cached client
# ============================================================
@lru_cache(maxsize=1)
def _get_gemini_client():
    """Create the Gemini client once per process."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Create a .env file in the project root "
            "with GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


# ============================================================
# Prompt construction
# ============================================================
_SYSTEM_INSTRUCTION = """You are a contract risk analyst helping a non-lawyer understand legal contracts.

When given a clause that has been flagged as risky, you:
1. Explain in plain English WHY the clause is risky (2-3 sentences max).
2. Suggest a concrete action the reader can take (1-2 sentences).

Rules:
- Use simple language. If you must use legal terms, define them briefly.
- Be specific: quote or paraphrase the problematic wording.
- Do not use hedging language ("may", "could", "might") unless genuinely uncertain.
- Compare against what standard practice looks like.
- Never invent facts not present in the clause or the risk signals.
"""


def _build_prompt(
    clause_text: str,
    label: Optional[str],
    risk_level: str,
    risk_score: float,
    risk_categories: List[str],
    explanation_hint: str,
    risky_language_tags: List[str],
) -> str:
    """Assemble the user prompt from all available signals."""

    parts = [
        "A contract clause has been flagged for review.",
        "",
        f"**Clause type:** {label or 'Unclassified'}",
        f"**Risk level:** {risk_level.upper()} (score {risk_score:.2f})",
    ]

    if risk_categories:
        parts.append(f"**Risk categories:** {', '.join(risk_categories)}")

    if risky_language_tags:
        parts.append(f"**Flagged wording patterns:** {', '.join(risky_language_tags)}")

    parts.append(f"**Automated analysis:** {explanation_hint}")
    parts.append("")
    parts.append("**Clause text:**")
    parts.append('"""')
    parts.append(clause_text.strip())
    parts.append('"""')
    parts.append("")
    parts.append(
        "Now provide your response in exactly this format:\n"
        "\n"
        "EXPLANATION:\n"
        "<your 2-3 sentence explanation here>\n"
        "\n"
        "SUGGESTED ACTION:\n"
        "<your 1-2 sentence recommendation here>"
    )
    return "\n".join(parts)


def _parse_response(text: str) -> Dict[str, str]:
    """
    Parse Gemini's response into explanation + suggested_action.
    Handles a few common format variations.
    """
    text = text.strip()

    explanation = ""
    suggested_action = ""

    # Try structured parse first
    if "EXPLANATION:" in text and "SUGGESTED ACTION:" in text:
        after_expl = text.split("EXPLANATION:", 1)[1]
        expl_part, action_part = after_expl.split("SUGGESTED ACTION:", 1)
        explanation = expl_part.strip()
        suggested_action = action_part.strip()
    else:
        # Fallback: treat whole text as explanation
        explanation = text

    return {
        "explanation": explanation,
        "suggested_action": suggested_action,
    }


# ============================================================
# Engine
# ============================================================
class ExplanationEngine:
    def __init__(self):
        cfg = get_config()["rag"]
        self.provider = cfg.get("llm_provider", "gemini")
        self.primary_model = cfg["primary_model"]
        self.fallback_models = list(cfg.get("fallback_models", []))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 400))
        self.timeout_sec = int(cfg.get("request_timeout_sec", 30))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.explain_at_or_above = cfg.get("explain_at_or_above", "medium")
        self.thinking_budget = cfg.get("thinking_budget", None)

        self._all_models = [self.primary_model] + self.fallback_models
        self.client = _get_gemini_client()

    # ---------------- helpers ----------------
    def _should_explain(self, risk_level: str) -> bool:
        order = {"low": 0, "medium": 1, "high": 2}
        return order.get(risk_level, 0) >= order.get(self.explain_at_or_above, 1)

    def _call_model(self, model_name: str, prompt: str) -> str:
        """
        Call one model. Retries up to self.max_retries on transient errors.
        Raises on permanent failure.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                gen_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "system_instruction": _SYSTEM_INSTRUCTION,
                }
                # Optional: disable extended thinking for faster, cheaper responses
                if self.thinking_budget is not None:
                    gen_config["thinking_config"] = {
                        "thinking_budget": self.thinking_budget,
                    }

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gen_config,
                )
                return (response.text or "").strip()
            
            except (genai_errors.ServerError, genai_errors.APIError) as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "code", "unknown")
                # 5xx or 429 → retryable; 4xx (except 429) → not retryable
                retryable = str(status).startswith("5") or "429" in str(status)
                if not retryable:
                    raise
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # exponential backoff: 1s, 2s
                    time.sleep(wait)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("unreachable")

    # ---------------- single clause ----------------
    def explain(
        self,
        clause,
        classification: Dict[str, Any],
        comparison: Dict[str, Any],
        risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a single clause.
        Always returns a dict — never raises on API failure.
        """
        risk_level = risk.get("risk_level", "low")

        # Skip LLM call for low-risk clauses
        if not self._should_explain(risk_level):
            return {
                "explanation": None,
                "suggested_action": None,
                "model_used": None,
                "latency_ms": 0,
                "skipped_reason": f"risk_level '{risk_level}' below threshold '{self.explain_at_or_above}'",
            }

        prompt = _build_prompt(
            clause_text=clause.text,
            label=classification.get("label"),
            risk_level=risk_level,
            risk_score=float(risk.get("risk_score", 0.0)),
            risk_categories=risk.get("risk_categories", []) or [],
            explanation_hint=risk.get("explanation_hint", ""),
            risky_language_tags=risk.get("signals", {}).get("risky_language_tags", []) or [],
        )

        # Try each model in order until one succeeds
        errors_seen = []
        for model_name in self._all_models:
            try:
                start = time.time()
                text = self._call_model(model_name, prompt)
                latency_ms = int((time.time() - start) * 1000)
                parsed = _parse_response(text)
                return {
                    "explanation": parsed["explanation"],
                    "suggested_action": parsed["suggested_action"],
                    "model_used": model_name,
                    "latency_ms": latency_ms,
                    "skipped_reason": None,
                }
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                errors_seen.append(f"{model_name}: {err}")
                continue

        # All models failed
        return {
            "explanation": None,
            "suggested_action": None,
            "model_used": None,
            "latency_ms": 0,
            "skipped_reason": f"all models failed: {'; '.join(errors_seen)}",
        }

    # ---------------- batch ----------------
    def explain_batch(
        self,
        clauses: List,
        classifications: List[Dict[str, Any]],
        comparisons: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch. Only calls the LLM for
        clauses at or above the configured threshold.
        """
        if not (len(clauses) == len(classifications) == len(comparisons) == len(risks)):
            raise ValueError("Length mismatch in explain_batch inputs")

        to_explain = [
            i for i, r in enumerate(risks)
            if self._should_explain(r.get("risk_level", "low"))
        ]

        if not to_explain:
            # Nothing to explain — return skip dicts for everyone
            return [
                {
                    "explanation": None,
                    "suggested_action": None,
                    "model_used": None,
                    "latency_ms": 0,
                    "skipped_reason": "risk level below threshold",
                }
                for _ in clauses
            ]

        iterator = to_explain
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(to_explain, desc="[explainer] generating", unit="clause")
            except ImportError:
                pass

        # Run explanations only for qualifying clauses
        results: List[Optional[Dict[str, Any]]] = [None] * len(clauses)
        for i in iterator:
            results[i] = self.explain(
                clauses[i], classifications[i], comparisons[i], risks[i]
            )

        # Fill in skips for the rest
        for i in range(len(clauses)):
            if results[i] is None:
                results[i] = {
                    "explanation": None,
                    "suggested_action": None,
                    "model_used": None,
                    "latency_ms": 0,
                    "skipped_reason": "risk level below threshold",
                }

        return results  # type: ignore