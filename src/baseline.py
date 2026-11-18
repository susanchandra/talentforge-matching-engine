"""
src/baseline.py — Keyword ATS baseline comparator for TalentForge.

Simulates a traditional ATS system that ranks candidates purely on
keyword overlap with the job description. Used as the comparison
baseline to demonstrate TalentForge's superiority.
"""

import re
from typing import List


# Common English stopwords to exclude from keyword matching
_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "of", "for", "to", "with",
    "at", "by", "from", "on", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her",
    "them", "my", "our", "your", "his", "their", "as", "if", "but",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "few", "more", "most", "other", "some", "such",
}


def _extract_keywords(text: str) -> List[str]:
    """
    Tokenize text into meaningful lowercase keywords, removing stopwords.

    Splits on whitespace and punctuation, lowercases all tokens,
    removes stopwords and single-character tokens.

    Args:
        text: Input text to tokenize.

    Returns:
        List of keyword strings.
    """
    # Split on whitespace and common punctuation
    tokens = re.split(r"[\s,\.;:!\?\(\)\[\]\{\}\"\'\/\\\-_]+", text.lower())
    # Filter: remove stopwords, empty strings, single chars
    keywords = [
        t for t in tokens
        if t and len(t) > 1 and t not in _STOPWORDS
    ]
    return keywords


def keyword_ats_rank(jd_text: str, candidates: List[dict]) -> List[dict]:
    """
    Rank candidates using simple keyword overlap — simulating a traditional ATS.

    This baseline demonstrates the core limitation of keyword-only matching:
    it rewards buzzword-stuffed resumes and misses strong candidates who
    describe their experience in non-standard terminology.

    Args:
        jd_text: Raw job description text.
        candidates: List of candidate dicts, each with keys:
                    - candidate_id (str)
                    - resume_text (str)

    Returns:
        List of result dicts sorted by keyword_score descending:
        [
          {
            "candidate_id": str,
            "keyword_score": float,       # matching_keywords / total_jd_keywords
            "matching_keywords_count": int,
            "matching_keywords": list[str]  # which keywords matched
          },
          ...
        ]
    """
    # Extract unique keywords from the JD
    jd_keywords = list(set(_extract_keywords(jd_text)))
    total_jd_keywords = len(jd_keywords)

    if total_jd_keywords == 0:
        # Edge case: JD has no extractable keywords
        return [
            {
                "candidate_id": c.get("candidate_id", c.get("id", "unknown")),
                "keyword_score": 0.0,
                "matching_keywords_count": 0,
                "matching_keywords": [],
            }
            for c in candidates
        ]

    results = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", candidate.get("id", "unknown"))
        resume_text = candidate.get("resume_text", "").lower()

        # Count which JD keywords appear anywhere in the resume
        matching = [kw for kw in jd_keywords if kw in resume_text]
        match_count = len(matching)
        score = match_count / total_jd_keywords

        results.append({
            "candidate_id": candidate_id,
            "keyword_score": round(score, 4),
            "matching_keywords_count": match_count,
            "matching_keywords": matching[:20],  # cap for readability
        })

    # Sort by score descending (stable sort preserves original order on ties)
    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results
