"""
Risk scoring engine — Phase 5.8
Combines classifier confidence + reference deviation + label agreement +
risky-language patterns + substantive-risk override into a per-clause
risk assessment.

Evolution:
  v1 (5.1): classifier confidence + deviation + mismatch
  v2 (5.5): + confidence floor, placeholder detection, label multipliers
  v3 (5.7): + risky-language lexicon (substantive signal)
  v4 (5.8): + substantive-risk override (strong signals floor the score)
"""
import re
from typing import List, Dict, Any, Optional, Pattern, Tuple
from dataclasses import dataclass, field

from ..utils.config import get_config


# ============================================================
# Label-to-category map
# ============================================================
_LEGAL_LABELS = {
    "Indemnifications", "Indemnity", "Liabilities", "Terminations",
    "Intellectual Property", "Warranties", "Representations",
    "Enforceability", "Binding Effects", "Assignments", "Assigns",
    "Specific Performance", "Remedies", "Enforcements", "Waiver Of Jury Trials",
    "Governing Laws", "Arbitration", "Jurisdictions", "Venues",
}
_FINANCIAL_LABELS = {
    "Payments", "Fees", "Expenses", "Costs", "Taxes", "Tax Withholdings",
    "Withholdings", "Base Salary", "Benefits", "Insurances",
    "Financial Statements", "Solvency", "Liens", "Interests",
}
_OPERATIONAL_LABELS = {
    "Notices", "Cooperation", "Duties", "Records", "Compliance With Laws",
    "Consents", "Approvals", "Authorizations", "Publicity",
    "Non-Disparagement", "Confidentiality", "Disclosures",
    "Effective Dates", "Employment", "Vacations", "Positions",
}


# ============================================================
# Data classes
# ============================================================
@dataclass
class RiskSignals:
    classifier_confidence: float
    deviation_from_reference: float
    classifier_exemplar_mismatch: float
    short_clause_penalty: float
    structural_skip: bool
    placeholder_clause: bool = False
    confidence_floor_hit: bool = False
    label_multiplier: float = 1.0
    risky_language_score: float = 0.0
    risky_language_tags: List[str] = field(default_factory=list)
    override_applied: str = ""   # "medium" | "high" | ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classifier_confidence": self.classifier_confidence,
            "deviation_from_reference": self.deviation_from_reference,
            "classifier_exemplar_mismatch": self.classifier_exemplar_mismatch,
            "short_clause_penalty": self.short_clause_penalty,
            "structural_skip": self.structural_skip,
            "placeholder_clause": self.placeholder_clause,
            "confidence_floor_hit": self.confidence_floor_hit,
            "label_multiplier": self.label_multiplier,
            "risky_language_score": self.risky_language_score,
            "risky_language_tags": self.risky_language_tags,
            "override_applied": self.override_applied,
        }


# ============================================================
# Risk Scorer
# ============================================================
class RiskScorer:
    def __init__(self):
        cfg = get_config()["risk"]
        self.thresholds = cfg["thresholds"]
        self.weights = cfg["weights"]
        self.categories = cfg["categories"]
        self.min_conf_for_risk = float(cfg.get("min_confidence_for_risk", 0.25))
        self.label_multipliers = dict(cfg.get("label_multipliers", {}) or {})
        self.override = dict(cfg.get("override", {}) or {})

        # Compile placeholder regexes
        self._placeholder_patterns: List[Pattern] = [
            re.compile(p) for p in (cfg.get("placeholder_patterns") or [])
        ]

        # Compile risky-lexicon patterns
        # Each entry: (compiled_pattern, weight, tag)
        self._risky_patterns: List[Tuple[Pattern, float, str]] = []
        for entry in (cfg.get("risky_lexicon") or []):
            try:
                self._risky_patterns.append((
                    re.compile(entry["pattern"], re.IGNORECASE),
                    float(entry["weight"]),
                    entry.get("tag", "unknown"),
                ))
            except (re.error, KeyError) as e:
                print(f"[scorer] Bad risky_lexicon entry {entry}: {e}")

        self.short_clause_chars = 50
        self.short_clause_max_penalty = 0.5

    # ---------------- helpers ----------------
    def _is_placeholder_clause(self, text: str) -> bool:
        if not text:
            return False
        placeholder_chars = 0
        for pat in self._placeholder_patterns:
            for m in pat.finditer(text):
                placeholder_chars += (m.end() - m.start())
        num_placeholders = sum(
            len(pat.findall(text)) for pat in self._placeholder_patterns
        )
        ratio = placeholder_chars / max(len(text), 1)
        return ratio > 0.15 or num_placeholders >= 2

    def _label_multiplier(self, label: Optional[str]) -> float:
        if not label:
            return 1.0
        return float(self.label_multipliers.get(label, 1.0))

    def _risky_language_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Scan text for risky-language patterns.

        Scoring model:
          - Take the max weight as the base.
          - Add 40% of each additional (unique-tag) weight on top.
          - Cap at 1.0.
        """
        if not text:
            return 0.0, []

        matched_weights: List[float] = []
        tags: List[str] = []

        for pat, weight, tag in self._risky_patterns:
            if pat.search(text):
                matched_weights.append(weight)
                tags.append(tag)

        if not matched_weights:
            return 0.0, []

        # Deduplicate tags
        seen = set()
        unique_tags = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        sorted_weights = sorted(matched_weights, reverse=True)
        base = sorted_weights[0]
        extra = sum(sorted_weights[1:]) * 0.4
        score = min(1.0, base + extra)
        return score, unique_tags

    # ---------------- Single clause ----------------
    def score(
        self,
        clause,
        classification: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:

        # --- Structural skip ---
        if classification.get("skipped", False):
            return {
                "risk_score": 0.0,
                "risk_level": "low",
                "risk_categories": [],
                "signals": RiskSignals(
                    classifier_confidence=1.0,
                    deviation_from_reference=0.0,
                    classifier_exemplar_mismatch=0.0,
                    short_clause_penalty=0.0,
                    structural_skip=True,
                ).to_dict(),
                "explanation_hint": (
                    f"Structural clause ({clause.structural_role}); "
                    "not scored for semantic risk."
                ),
            }

        conf = float(classification.get("confidence") or 0.0)
        deviation = float(comparison.get("deviation_score") or 0.0)
        mismatch_flag = comparison.get("agrees_with_classifier")
        mismatch = 0.0 if mismatch_flag in (True, None) else 1.0

        length = clause.char_end - clause.char_start
        if length < self.short_clause_chars:
            short_penalty = self.short_clause_max_penalty
        elif length < 100:
            short_penalty = self.short_clause_max_penalty * (
                1 - (length - self.short_clause_chars) / 50.0
            )
        else:
            short_penalty = 0.0

        is_placeholder = self._is_placeholder_clause(clause.text)
        confidence_floor_hit = conf < self.min_conf_for_risk

        # ---- Risky-language signal ----
        risky_score, risky_tags = self._risky_language_score(clause.text)

        # --- Normalize signals to "risk contribution" ---
        conf_risk = 1.0 - conf
        dev_risk = deviation
        mismatch_risk = mismatch

        # --- Composite weighted score ---
        # 4 signals: deviation, confidence-uncertainty, mismatch, risky-lang
        w = self.weights
        w_dev = w["deviation_from_reference"]
        w_conf = w["risky_pattern_score"]
        w_mismatch = w["entity_anomaly_score"]
        w_risky_lang = 0.40

        total = w_dev + w_conf + w_mismatch + w_risky_lang
        w_dev /= total
        w_conf /= total
        w_mismatch /= total
        w_risky_lang /= total

        composite = (
            w_dev * dev_risk
            + w_conf * conf_risk
            + w_mismatch * mismatch_risk
            + w_risky_lang * risky_score
        )

        if short_penalty > 0:
            composite = composite * (1.0 - short_penalty)
        if is_placeholder:
            composite = composite * 0.5

        if confidence_floor_hit and risky_score < 0.5:
            composite = min(composite, self.thresholds["low"] - 0.01)

        multiplier = self._label_multiplier(classification.get("label"))
        composite = composite * multiplier

        # --- Substantive-risk override ---
        # A strong risky-language signal floors the score, because
        # substantive risk should not be drowned out by high classifier
        # confidence or low deviation.
        override_cfg = self.override
        override_applied = ""
        if risky_score >= override_cfg.get("risky_lang_high_floor", 0.85):
            floor = override_cfg.get("high_score_floor", 0.70)
            if composite < floor:
                composite = floor
                override_applied = "high"
        elif risky_score >= override_cfg.get("risky_lang_medium_floor", 0.65):
            floor = override_cfg.get("medium_score_floor", 0.50)
            if composite < floor:
                composite = floor
                override_applied = "medium"

        composite = max(0.0, min(1.0, composite))

        # --- Risk level ---
        if composite >= self.thresholds["high"]:
            level = "high"
        elif composite >= self.thresholds["medium"]:
            level = "medium"
        else:
            level = "low"

        categories = self._categorize(
            label=classification.get("label"),
            risk_level=level,
            confidence=conf,
            deviation=deviation,
            mismatch=mismatch,
            is_placeholder=is_placeholder,
            risky_tags=risky_tags,
        )

        hint = self._explain_hint(
            label=classification.get("label"),
            level=level,
            confidence=conf,
            deviation=deviation,
            mismatch=mismatch,
            short=short_penalty > 0,
            is_placeholder=is_placeholder,
            confidence_floor_hit=confidence_floor_hit,
            risky_tags=risky_tags,
            override_applied=override_applied,
        )

        signals = RiskSignals(
            classifier_confidence=conf,
            deviation_from_reference=deviation,
            classifier_exemplar_mismatch=mismatch,
            short_clause_penalty=short_penalty,
            structural_skip=False,
            placeholder_clause=is_placeholder,
            confidence_floor_hit=confidence_floor_hit,
            label_multiplier=multiplier,
            risky_language_score=risky_score,
            risky_language_tags=risky_tags,
            override_applied=override_applied,
        )

        return {
            "risk_score": round(composite, 4),
            "risk_level": level,
            "risk_categories": categories,
            "signals": signals.to_dict(),
            "explanation_hint": hint,
        }

    # ---------------- Batch ----------------
    def score_batch(
        self,
        clauses: List,
        classifications: List[Dict[str, Any]],
        comparisons: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not (len(clauses) == len(classifications) == len(comparisons)):
            raise ValueError("Length mismatch in score_batch inputs")
        return [
            self.score(c, cls, cmp)
            for c, cls, cmp in zip(clauses, classifications, comparisons)
        ]

    # ---------------- Category logic ----------------
    def _categorize(
        self,
        label: Optional[str],
        risk_level: str,
        confidence: float,
        deviation: float,
        mismatch: float,
        is_placeholder: bool,
        risky_tags: List[str],
    ) -> List[str]:
        categories: List[str] = []
        if not label:
            return categories

        if risk_level in ("medium", "high"):
            if label in _LEGAL_LABELS:
                categories.append("Legal")
            if label in _FINANCIAL_LABELS:
                categories.append("Financial")
            if label in _OPERATIONAL_LABELS:
                categories.append("Operational")

        tag_to_category = {
            "unilateral-discretion": "Operational",
            "unilateral-timing": "Operational",
            "no-cause": "Legal",
            "no-notice": "Operational",
            "no-liability": "Legal",
            "unlimited-scope": "Legal",
            "perpetual": "Legal",
            "irrevocable": "Legal",
            "waiver": "Legal",
            "auto-renew": "Operational",
            "non-compete": "Operational",
            "non-solicit": "Operational",
            "assignment": "Financial",
            "uncompensated": "Financial",
            "immediate": "Operational",
            "indemnify": "Legal",
        }
        for tag in risky_tags:
            cat = tag_to_category.get(tag)
            if cat and cat not in categories:
                categories.append(cat)

        is_ambiguous = (
            (confidence < 0.35 and mismatch > 0)
            or (deviation > 0.55 and mismatch > 0)
        )
        if is_ambiguous and not is_placeholder:
            categories.append("Ambiguity")

        seen = set()
        unique = []
        for c in categories:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    # ---------------- Explanation hints ----------------
    def _explain_hint(
        self,
        label: Optional[str],
        level: str,
        confidence: float,
        deviation: float,
        mismatch: float,
        short: bool,
        is_placeholder: bool,
        confidence_floor_hit: bool,
        risky_tags: List[str],
        override_applied: str,
    ) -> str:
        parts = []

        # --- Base description ---
        if label:
            parts.append(f"'{label}' clause")
        else:
            parts.append("Unlabeled clause")

        # --- Risk level ---
        if level == "high":
            parts.append("HIGH RISK")
        elif level == "medium":
            parts.append("moderate risk")
        # "low" adds nothing — keeps hints shorter

        # --- Risky language tags ---
        if risky_tags:
            parts.append(f"risky language: {', '.join(risky_tags)}")

        # --- Override notice ---
        if override_applied:
            parts.append(
                f"score floored by substantive-risk override ({override_applied})"
            )

                # --- Supplementary reasons ---
        # Decide which "why" narrative fits this clause:
        #   A. Form-field (placeholder clause) — special message
        #   B. Override applied — mention classifier uncertainty if notable
        #   C. Floor-capped without risky language — say so explicitly
        #   D. Everything else — spell out contributing signals

        if is_placeholder:
            parts.append("form-field clause (mostly placeholders)")
        elif override_applied:
            if confidence < 0.5:
                parts.append(f"classifier uncertain (conf={confidence:.0%})")
        elif confidence_floor_hit and not risky_tags:
            # Capped because: low confidence AND no risky language found
            parts.append(
                f"classifier uncertain (conf={confidence:.0%}) — "
                "no risky language detected"
            )
        else:
            # Normal path — spell out signals
            if confidence < 0.5:
                parts.append(f"classifier uncertain (conf={confidence:.0%})")
            if deviation > 0.55:
                parts.append(
                    f"significantly different from standard "
                    f"({deviation:.0%} deviation)"
                )
            elif deviation > 0.35:
                parts.append(
                    f"somewhat different from standard ({deviation:.0%} deviation)"
                )
            if mismatch > 0:
                parts.append("classifier and reference-bank disagree")
            if short:
                parts.append("short clause (fragment)")

        return " — ".join(parts)        