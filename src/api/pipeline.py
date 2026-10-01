"""
Pipeline orchestrator — Phase 7.2
Chains all phases (ingestion → segmentation → classification →
comparison → scoring → explanation) into a single function.

The API endpoint calls analyze_document(path) and gets back the full
AnalysisResponse. This keeps main.py thin.
"""
from pathlib import Path
from typing import Optional

from ..ingestion import ingest
from ..segmentation import segment
from ..nlp import ClauseClassifier
from ..risk import ClauseComparator, RiskScorer
from ..rag import ExplanationEngine
from .schemas import (
    AnalysisResponse, DocumentInfo, SummaryStats,
    ClauseResult, ClassificationInfo, RiskInfo, ExplanationInfo,
)


# Cache expensive objects at module level — loaded once per process
_classifier: Optional[ClauseClassifier] = None
_comparator: Optional[ClauseComparator] = None
_scorer: Optional[RiskScorer] = None
_explainer: Optional[ExplanationEngine] = None


def _get_classifier() -> ClauseClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ClauseClassifier()
    return _classifier


def _get_comparator() -> ClauseComparator:
    global _comparator
    if _comparator is None:
        _comparator = ClauseComparator()
    return _comparator


def _get_scorer() -> RiskScorer:
    global _scorer
    if _scorer is None:
        _scorer = RiskScorer()
    return _scorer


def _get_explainer() -> ExplanationEngine:
    global _explainer
    if _explainer is None:
        _explainer = ExplanationEngine()
    return _explainer


def analyze_document(
    file_path: Path,
    generate_explanations: bool = True,
    show_progress: bool = False,
) -> AnalysisResponse:
    """
    Run the entire pipeline on a single document.

    Args:
        file_path: Path to the contract file (PDF / DOCX / TXT).
        generate_explanations: If False, skip the LLM layer (faster, no cost).
        show_progress: Print tqdm bars for long steps.

    Returns:
        AnalysisResponse with all clauses, scores, and explanations.
    """
    # 1. Ingest
    doc = ingest(file_path)

    # 2. Segment
    clauses = segment(doc.full_text)

    # 3. Classify
    clf = _get_classifier()
    classifications = clf.classify_clauses(clauses, top_k=3, show_progress=show_progress)

    # 4. Compare against reference bank
    cmp = _get_comparator()
    comparisons = cmp.compare_batch(clauses, predictions=classifications, show_progress=show_progress)

    # 5. Score risk
    scorer = _get_scorer()
    risks = scorer.score_batch(clauses, classifications, comparisons)

    # 6. Generate explanations (only for qualifying clauses)
    if generate_explanations:
        explainer = _get_explainer()
        explanations = explainer.explain_batch(
            clauses, classifications, comparisons, risks,
            show_progress=show_progress,
        )
    else:
        explanations = [
            {"explanation": None, "suggested_action": None,
             "model_used": None, "latency_ms": 0,
             "skipped_reason": "explanations disabled by request"}
            for _ in clauses
        ]

    # 7. Build response
    clause_results = []
    for c, cls, cm, rk, ex in zip(clauses, classifications, comparisons, risks, explanations):
        clause_results.append(ClauseResult(
            id=c.id,
            number=c.number,
            structural_role=c.structural_role,
            level=c.level,
            parent_id=c.parent_id,
            text=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            classification=ClassificationInfo(
                label=cls.get("label"),
                confidence=float(cls.get("confidence") or 0.0),
                top_k=[
                    {"label": lbl, "probability": float(prob)}
                    for lbl, prob in (cls.get("top_k") or [])
                ],
            ),
            risk=RiskInfo(
                score=float(rk["risk_score"]),
                level=rk["risk_level"],
                categories=rk.get("risk_categories", []),
                explanation_hint=rk.get("explanation_hint", ""),
                signals=rk.get("signals", {}),
            ),
            explanation=ExplanationInfo(
                text=ex.get("explanation"),
                suggested_action=ex.get("suggested_action"),
                model_used=ex.get("model_used"),
                latency_ms=int(ex.get("latency_ms", 0)),
                skipped_reason=ex.get("skipped_reason"),
            ),
        ))

    # Summary stats
    substantive = [r for r in clause_results if r.structural_role not in
                   {"PARTIES", "RECITAL", "SIGNATURE", "PREAMBLE"}]
    summary = SummaryStats(
        total_clauses=len(clause_results),
        high=sum(1 for r in substantive if r.risk.level == "high"),
        medium=sum(1 for r in substantive if r.risk.level == "medium"),
        low=sum(1 for r in substantive if r.risk.level == "low"),
        skipped_structural=len(clause_results) - len(substantive),
    )

    return AnalysisResponse(
        document=DocumentInfo(
            filename=file_path.name,
            format=doc.format,
            pages=len(doc.pages),
            chars=len(doc.full_text),
        ),
        summary=summary,
        clauses=clause_results,
    )