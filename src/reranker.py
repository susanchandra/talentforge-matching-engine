"""
src/reranker.py — Cross-encoder reranking for TalentForge.

Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to produce a precise
relevance score for each (JD, resume) pair, then blends with the
hybrid retrieval score for the final ranking.
"""

from typing import List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidates using a cross-encoder relevance model.

    The cross-encoder scores each (query, passage) pair jointly,
    producing more accurate relevance signals than bi-encoder cosine
    similarity alone — at the cost of O(n) forward passes.

    We use it on the top-20 retrieved candidates only (post-retrieval
    reranking) to keep latency reasonable.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        """Load the cross-encoder model (downloads on first use)."""
        print("Loading reranker model (first time only)...")
        self.model = CrossEncoder(self.MODEL_NAME, max_length=512)
        print(f"CrossEncoderReranker loaded: {self.MODEL_NAME}")

    def rerank(
        self,
        jd_text: str,
        candidates: List[dict],
        top_k: int = 10,
    ) -> List[dict]:
        """
        Rerank candidates using cross-encoder scores blended with hybrid scores.

        Args:
            jd_text: Raw job description text (used as the query).
            candidates: List of candidate dicts, each with keys:
                        - candidate_id (str)
                        - resume_text (str)
                        - hybrid_score (float, 0-1)
            top_k: Number of final results to return.

        Returns:
            List of top_k dicts sorted by final_score descending:
            [
              {
                "candidate_id": str,
                "hybrid_score": float,
                "cross_encoder_score": float,    # raw logit, not normalized
                "cross_encoder_score_norm": float, # min-max normalized 0-1
                "final_score": float,             # blended score
              },
              ...
            ]
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        # Truncate resume to first 1000 chars to fit within 512 tokens
        pairs = [
            (jd_text[:500], c.get("resume_text", "")[:1000])
            for c in candidates
        ]

        # Score all pairs in one batch
        raw_scores = self.model.predict(pairs, show_progress_bar=False).tolist()

        # Min-max normalize cross-encoder scores to [0, 1]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        if score_range < 1e-9:
            # All scores identical — normalize to 0.5
            normalized_scores = [0.5] * len(raw_scores)
        else:
            normalized_scores = [
                (s - min_score) / score_range for s in raw_scores
            ]

        # Blend: 60% hybrid (retrieval) + 40% cross-encoder (rerank)
        results = []
        for candidate, raw_ce, norm_ce in zip(candidates, raw_scores, normalized_scores):
            hybrid = candidate.get("hybrid_score", 0.0)
            final = 0.6 * hybrid + 0.4 * norm_ce

            results.append({
                "candidate_id": candidate["candidate_id"],
                "hybrid_score": round(hybrid, 4),
                "cross_encoder_score": round(float(raw_ce), 4),
                "cross_encoder_score_norm": round(norm_ce, 4),
                "final_score": round(final, 4),
            })

        # Sort by final_score descending and return top_k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
