"""
src/reranker.py — Cross-encoder reranking for TalentForge.

Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to produce a precise
relevance score for each (JD, resume) pair, then blends with the
hybrid retrieval score for the final ranking.
"""

from typing import List, Optional
import numpy as np


class CrossEncoderReranker:
    """
    Reranks candidates using a cross-encoder relevance model.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        """Load tokenizer and cross-encoder model."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print(f"Loading CrossEncoder: {self.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        print(f"CrossEncoderReranker loaded successfully.")

    def rerank(
        self,
        jd_text: str,
        candidates: List[dict],
        top_k: int = 10,
    ) -> List[dict]:
        """
        Rerank candidates using cross-encoder scores blended with hybrid scores.

        Args:
            jd_text: Raw job description text (used as query).
            candidates: List of candidate dicts with candidate_id, resume_text, hybrid_score.
            top_k: Number of final results to return.

        Returns:
            List of candidate dicts with final_score, cross_encoder_score, etc.
        """
        if not candidates:
            return []

        import torch

        # Prepare pairs: JD truncated to 500 chars, resume to 1000 chars
        pairs = [
            (jd_text[:500], c.get("resume_text", "")[:1000])
            for c in candidates
        ]

        # Batch tokenization
        features = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**features).logits
            # Binary classification or single logit score
            if logits.shape[1] == 1:
                raw_scores = logits.squeeze(-1).cpu().numpy().tolist()
            else:
                raw_scores = logits[:, 1].cpu().numpy().tolist()

        # Min-max normalization across retrieved candidates
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        span = max_s - min_s

        if span < 1e-9:
            norm_scores = [0.5] * len(raw_scores)
        else:
            norm_scores = [(s - min_s) / span for s in raw_scores]

        # Blend: 60% hybrid score + 40% cross-encoder score
        results = []
        for candidate, raw_ce, norm_ce in zip(candidates, raw_scores, norm_scores):
            hybrid = candidate.get("hybrid_score", 0.0)
            final = 0.60 * hybrid + 0.40 * norm_ce

            results.append({
                "candidate_id": candidate["candidate_id"],
                "hybrid_score": round(hybrid, 4),
                "cross_encoder_score": round(float(raw_ce), 4),
                "cross_encoder_score_norm": round(norm_ce, 4),
                "final_score": round(final, 4),
            })

        # Sort descending by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
