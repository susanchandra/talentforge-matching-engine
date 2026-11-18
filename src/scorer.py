"""
src/scorer.py — 4-dimensional structured candidate scoring for TalentForge.

Computes dimension scores for skill match, experience depth,
project relevance, and seniority fit — without any LLM calls.
All scoring is deterministic and explainable.
"""

from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Seniority level mappings
# ──────────────────────────────────────────────────────────────────────────────

SENIORITY_ORDER = ["junior", "mid", "senior", "staff", "principal"]
SENIORITY_INDEX = {level: i for i, level in enumerate(SENIORITY_ORDER)}

# Experience year ranges that define each seniority level
SENIORITY_YEAR_RANGES = {
    "junior": (0, 2),
    "mid": (2, 5),
    "senior": (5, 9),
    "staff": (9, 14),
    "principal": (14, 99),
}

# Scoring weights for the final structured_score
DIMENSION_WEIGHTS = {
    "skill_match": 0.35,
    "experience_depth": 0.25,
    "project_relevance": 0.25,
    "seniority_fit": 0.15,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_skill(skill: str) -> List[str]:
    """Tokenize a skill string into lowercase tokens for fuzzy matching."""
    import re
    return re.split(r"[\s\-_/\+\.]+", skill.lower())


def _skills_fuzzy_match(required_skill: str, candidate_skills: List[str]) -> bool:
    """
    Check if any candidate skill is a fuzzy match for the required skill.

    Uses substring matching and token overlap — no external library needed.
    A match occurs if:
    - The required skill is a substring of a candidate skill (or vice versa), OR
    - The required skill tokens overlap significantly with a candidate skill's tokens.

    Args:
        required_skill: Skill from the JD requirements.
        candidate_skills: List of skills from the candidate profile.

    Returns:
        True if a match is found, False otherwise.
    """
    req_lower = required_skill.lower().strip()
    req_tokens = set(_normalize_skill(req_lower))

    for cand_skill in candidate_skills:
        cand_lower = cand_skill.lower().strip()
        cand_tokens = set(_normalize_skill(cand_lower))

        # Direct substring containment
        if req_lower in cand_lower or cand_lower in req_lower:
            return True

        # Token overlap: if majority of required tokens appear in candidate skill tokens
        if req_tokens and cand_tokens:
            overlap = req_tokens & cand_tokens
            overlap_ratio = len(overlap) / len(req_tokens)
            if overlap_ratio >= 0.6:  # 60% token overlap = match
                return True

    return False


def _experience_years_to_seniority(years: float) -> str:
    """Map years of experience to a seniority level string."""
    if years < 2:
        return "junior"
    elif years < 5:
        return "mid"
    elif years < 9:
        return "senior"
    elif years < 14:
        return "staff"
    else:
        return "principal"


def _seniority_distance(level_a: str, level_b: str) -> int:
    """
    Compute the ordinal distance between two seniority levels.

    Returns 0 for exact match, 1 for adjacent, etc.
    Handles unknown levels gracefully.
    """
    idx_a = SENIORITY_INDEX.get(level_a, 2)  # default to mid
    idx_b = SENIORITY_INDEX.get(level_b, 2)
    return abs(idx_a - idx_b)


def _project_keyword_overlap(project_desc: str, jd_keywords: List[str]) -> float:
    """
    Compute token overlap between a project description and JD keywords.

    Args:
        project_desc: Project description + technologies text.
        jd_keywords: List of keywords from required_skills + key_expectations.

    Returns:
        Overlap score between 0.0 and 1.0.
    """
    if not jd_keywords:
        return 0.0

    desc_lower = project_desc.lower()
    matches = sum(1 for kw in jd_keywords if kw.lower() in desc_lower)
    return min(1.0, matches / len(jd_keywords))


# ──────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ──────────────────────────────────────────────────────────────────────────────

class StructuredScorer:
    """
    Computes 4-dimensional structured candidate-JD match scores.

    All scoring is deterministic (no LLM calls) for speed and reproducibility.
    Dimension scores are blended into a weighted structured_score.
    """

    def score_candidate(self, jd_schema: dict, candidate_schema: dict) -> dict:
        """
        Compute all four dimension scores and the blended structured_score.

        Args:
            jd_schema: Extracted JD schema dict (from extractor.extract_jd_schema).
            candidate_schema: Extracted candidate schema dict (from extractor.extract_candidate_schema).

        Returns:
            Dict with keys:
            - skill_match (float, 0-1)
            - experience_depth (float, 0-1)
            - project_relevance (float, 0-1)
            - seniority_fit (float, 0-1)
            - structured_score (float, 0-1) — weighted average
        """
        skill_match = self._score_skill_match(jd_schema, candidate_schema)
        experience_depth = self._score_experience_depth(jd_schema, candidate_schema)
        project_relevance = self._score_project_relevance(jd_schema, candidate_schema)
        seniority_fit = self._score_seniority_fit(jd_schema, candidate_schema)

        structured_score = (
            DIMENSION_WEIGHTS["skill_match"] * skill_match
            + DIMENSION_WEIGHTS["experience_depth"] * experience_depth
            + DIMENSION_WEIGHTS["project_relevance"] * project_relevance
            + DIMENSION_WEIGHTS["seniority_fit"] * seniority_fit
        )

        return {
            "skill_match": round(skill_match, 4),
            "experience_depth": round(experience_depth, 4),
            "project_relevance": round(project_relevance, 4),
            "seniority_fit": round(seniority_fit, 4),
            "structured_score": round(structured_score, 4),
        }

    # ── Dimension 1: Skill Match ──────────────────────────────────────────────

    def _score_skill_match(self, jd_schema: dict, candidate_schema: dict) -> float:
        """
        Score how many of the JD's required skills the candidate possesses.

        Uses fuzzy token matching so "PyTorch" matches "pytorch" and
        "machine learning" matches "ML" via substring overlap.
        """
        required_skills: List[str] = jd_schema.get("required_skills", [])
        candidate_skills: List[str] = candidate_schema.get("skills", [])

        if not required_skills:
            return 0.5  # No requirements specified — neutral score

        matched = sum(
            1 for req in required_skills
            if _skills_fuzzy_match(req, candidate_skills)
        )

        return min(1.0, matched / len(required_skills))

    # ── Dimension 2: Experience Depth ─────────────────────────────────────────

    def _score_experience_depth(self, jd_schema: dict, candidate_schema: dict) -> float:
        """
        Score how well the candidate's years of experience match the JD's seniority requirement.

        Maps years → seniority level, then computes level-distance score.
        """
        required_level = jd_schema.get("seniority_level", "mid").lower()
        experience_years = float(candidate_schema.get("experience_years", 0) or 0)
        candidate_level = _experience_years_to_seniority(experience_years)

        distance = _seniority_distance(required_level, candidate_level)

        # Score table based on level distance
        score_map = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2}
        return score_map.get(distance, 0.1)

    # ── Dimension 3: Project Relevance ────────────────────────────────────────

    def _score_project_relevance(self, jd_schema: dict, candidate_schema: dict) -> float:
        """
        Score the relevance of candidate projects to the JD using token overlap.

        Combines required_skills and key_expectations as the reference keyword set.
        Averages overlap scores across all candidate projects.
        """
        required_skills: List[str] = jd_schema.get("required_skills", [])
        key_expectations: List[str] = jd_schema.get("key_expectations", [])
        # Build keyword reference set from JD
        jd_keywords = list(set(required_skills + key_expectations))

        projects: List[dict] = candidate_schema.get("projects", [])
        if not projects:
            return 0.1  # No projects is a weak signal

        scores = []
        for project in projects:
            # Combine project description and technologies into one text blob
            desc = project.get("description", "")
            techs = " ".join(project.get("technologies", []))
            project_text = f"{desc} {techs}"
            score = _project_keyword_overlap(project_text, jd_keywords)
            scores.append(score)

        return round(sum(scores) / len(scores), 4)

    # ── Dimension 4: Seniority Fit ────────────────────────────────────────────

    def _score_seniority_fit(self, jd_schema: dict, candidate_schema: dict) -> float:
        """
        Score the direct seniority level match between JD requirement and
        the candidate's most recent role seniority.
        """
        required_level = jd_schema.get("seniority_level", "mid").lower()

        role_history: List[dict] = candidate_schema.get("role_history", [])
        if not role_history:
            return 0.3  # No role history — low confidence

        # Use the most recent role (last in list)
        most_recent_role = role_history[-1]
        candidate_level = most_recent_role.get("seniority_level", "mid").lower()

        distance = _seniority_distance(required_level, candidate_level)

        # Generous scoring: exact=1.0, one off=0.7, two=0.4, three+=0.1
        score_map = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
        return score_map.get(distance, 0.1)
