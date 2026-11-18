"""
src/evaluator.py — NDCG and Precision@K evaluation metrics for TalentForge.

Compares TalentForge hybrid ranking vs keyword ATS baseline using
information retrieval metrics grounded in ground-truth quality labels.
"""

import math
from typing import List


# Relevance grade mapping from ground truth quality labels
RELEVANCE_GRADES = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "unknown": 0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_ndcg_at_k(
    ranked_ids: List[str],
    ground_truth_quality: dict,
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k (NDCG@k).

    Args:
        ranked_ids: Ordered list of candidate IDs from top to bottom rank.
        ground_truth_quality: Dict mapping candidate_id -> "high"/"medium"/"low".
        k: Cutoff rank.

    Returns:
        NDCG@k score (0.0 to 1.0).
    """
    # Get relevance grades for ranked list (up to k)
    top_k_ids = ranked_ids[:k]
    grades = [
        RELEVANCE_GRADES.get(ground_truth_quality.get(cid, "unknown"), 0)
        for cid in top_k_ids
    ]

    # Compute DCG@k
    dcg = sum(
        grade / math.log2(rank + 2)  # rank+2 because log2(1)=0, rank is 0-indexed
        for rank, grade in enumerate(grades)
    )

    # Compute IDCG@k (ideal DCG using best possible ranking)
    all_grades = sorted(
        [RELEVANCE_GRADES.get(q, 0) for q in ground_truth_quality.values()],
        reverse=True,
    )
    idcg = sum(
        grade / math.log2(rank + 2)
        for rank, grade in enumerate(all_grades[:k])
        if grade > 0
    )

    if idcg < 1e-9:
        return 0.0

    return round(dcg / idcg, 4)


def compute_precision_at_k(
    ranked_ids: List[str],
    ground_truth_quality: dict,
    k: int = 10,
) -> float:
    """
    Compute Precision@K — fraction of top-k results that are "high" quality.

    Args:
        ranked_ids: Ordered list of candidate IDs.
        ground_truth_quality: Dict mapping candidate_id -> quality label.
        k: Cutoff rank.

    Returns:
        Precision@k (0.0 to 1.0).
    """
    top_k_ids = ranked_ids[:k]
    high_quality_count = sum(
        1 for cid in top_k_ids
        if ground_truth_quality.get(cid, "unknown") == "high"
    )
    return round(high_quality_count / k, 4) if k > 0 else 0.0


def run_evaluation(
    pipeline_results: List[dict],
    baseline_results: List[dict],
    candidates_meta: dict,
    k: int = 10,
) -> dict:
    """
    Run full evaluation comparing TalentForge vs keyword ATS baseline.

    Args:
        pipeline_results: TalentForge ranked list of dicts with 'candidate_id'.
        baseline_results: ATS baseline ranked list of dicts with 'candidate_id'.
        candidates_meta: Dict mapping candidate_id -> candidate metadata dict
                         (must include 'ground_truth_quality' field).
        k: Evaluation cutoff rank.

    Returns:
        Comparison dict:
        {
          pipeline_ndcg: float,
          baseline_ndcg: float,
          pipeline_precision: float,
          baseline_precision: float,
          ndcg_improvement_pct: float,
          precision_improvement_pct: float,
        }
    """
    # Build ground truth quality mapping
    ground_truth = {
        cid: meta.get("ground_truth_quality", "unknown")
        for cid, meta in candidates_meta.items()
    }

    # Extract ranked ID lists
    pipeline_ids = [r["candidate_id"] for r in pipeline_results]
    baseline_ids = [r["candidate_id"] for r in baseline_results]

    # Compute metrics
    pipeline_ndcg = compute_ndcg_at_k(pipeline_ids, ground_truth, k)
    baseline_ndcg = compute_ndcg_at_k(baseline_ids, ground_truth, k)
    pipeline_precision = compute_precision_at_k(pipeline_ids, ground_truth, k)
    baseline_precision = compute_precision_at_k(baseline_ids, ground_truth, k)

    # Compute improvement percentages
    def _improvement_pct(pipeline_val: float, baseline_val: float) -> float:
        if baseline_val < 1e-9:
            return 100.0 if pipeline_val > 0 else 0.0
        return round((pipeline_val - baseline_val) / baseline_val * 100, 2)

    return {
        "pipeline_ndcg": pipeline_ndcg,
        "baseline_ndcg": baseline_ndcg,
        "pipeline_precision": pipeline_precision,
        "baseline_precision": baseline_precision,
        "ndcg_improvement_pct": _improvement_pct(pipeline_ndcg, baseline_ndcg),
        "precision_improvement_pct": _improvement_pct(pipeline_precision, baseline_precision),
    }
