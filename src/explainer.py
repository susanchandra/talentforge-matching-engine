"""
src/explainer.py — LLM-powered explanation card generation for TalentForge.

This is the core differentiator: structured, evidence-based explanations
that tell recruiters WHY a candidate was ranked the way they were.
"""

import json
import re
from typing import Optional

import openai


# ──────────────────────────────────────────────────────────────────────────────
# Mock fallback (used when no API key is present)
# ──────────────────────────────────────────────────────────────────────────────

MOCK_EXPLANATION_CARD = {
    "summary": "Strong candidate with relevant technical background.",
    "top_strengths": [
        "Technical skills align with role requirements.",
        "Demonstrated project experience in the relevant domain.",
        "Career progression shows consistent growth.",
    ],
    "concern": "Insufficient signal to assess leadership experience.",
    "interview_probe": "Describe a time you owned a project end-to-end.",
    "recommendation": "maybe",
    "ats_miss": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM_PROMPT = """You are a senior technical recruiter with 10 years of experience.
You write hiring recommendations that are specific, evidence-based,
and actionable. Never use generic phrases like "strong candidate"
or "good fit" without citing specific evidence from the resume.

Your task: write a structured explanation card for a candidate
who has been ranked by an AI hiring system.

Rules:
1. Every strength claim must cite specific evidence from the resume
   (project name, company name, technology, impact number if available).
2. Include exactly one concern or gap — do not skip this even for
   top-ranked candidates. Concerns make explanations credible.
3. End with a specific interview probe question targeting the concern.
4. Vary your language — do not use the same sentence structure
   across different candidates.
5. Write as if you are advising a hiring manager, not summarizing a resume.
6. Include the "ats_miss" field only if the candidate would likely
   rank poorly on keyword matching despite being a strong candidate
   — otherwise set to null.

Return ONLY valid JSON with no preamble or markdown:
{
  "summary": "2-3 sentence recruiter assessment. Specific, evidence-based.",
  "top_strengths": [
    "strength 1 with specific evidence",
    "strength 2 with specific evidence",
    "strength 3 with specific evidence"
  ],
  "concern": "one specific gap or uncertainty, not generic",
  "interview_probe": "one specific question to probe the concern",
  "recommendation": "one of: strong yes, yes, maybe, no",
  "ats_miss": "why ATS would have missed this person, or null"
}"""

RECRUITER_MEMO_SYSTEM_PROMPT = """You are a senior recruiting manager summarizing AI-assisted candidate
screening results for a hiring team. Write a concise, professional
briefing memo (200-250 words). Cover:
1. The top 3 recommended candidates and why (one sentence each).
2. One pattern you noticed across the shortlist.
3. One caveat or limitation of the screening the team should know.

Write in memo format: plain paragraphs, no bullet points.
Do not use the phrase "AI system" — use "the screening" instead."""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json...``` wrappers from LLM output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


class ExplanationEngine:
    """
    Generates structured explanation cards and recruiter memos using GPT-4o-mini.

    If the OpenAI client is None or the API call fails, falls back to a
    safe mock explanation so the app always renders.
    """

    def __init__(self, openai_client: Optional[openai.OpenAI]):
        """
        Args:
            openai_client: Initialized OpenAI client, or None for mock mode.
        """
        self.client = openai_client
        self._mock_mode = openai_client is None

    def generate_explanation_card(
        self,
        jd_schema: dict,
        candidate_schema: dict,
        scores: dict,
        trajectory: dict,
        rank: int,
    ) -> dict:
        """
        Generate a structured explanation card for a ranked candidate.

        Args:
            jd_schema: Extracted JD schema dict.
            candidate_schema: Extracted candidate schema dict.
            scores: Dimension scores dict from StructuredScorer.
            trajectory: Trajectory analysis dict from TrajectoryAnalyzer.
            rank: Candidate's rank position (1-indexed).

        Returns:
            Explanation card dict with: summary, top_strengths, concern,
            interview_probe, recommendation, ats_miss.
        """
        if self._mock_mode:
            card = MOCK_EXPLANATION_CARD.copy()
            card["summary"] = (
                f"{candidate_schema.get('name', 'This candidate')} brings "
                f"{candidate_schema.get('experience_years', 'several')} years of experience "
                f"with demonstrated skills in {', '.join(candidate_schema.get('skills', ['the required domain'])[:3])}."
            )
            return card

        user_message = (
            f"Candidate rank: {rank}\n"
            f"JD: {json.dumps(jd_schema)}\n"
            f"Candidate: {json.dumps(candidate_schema)}\n"
            f"Scores: skill_match={scores.get('skill_match', 0):.2f}, "
            f"experience_depth={scores.get('experience_depth', 0):.2f}, "
            f"project_relevance={scores.get('project_relevance', 0):.2f}, "
            f"trajectory={trajectory.get('trajectory_class', 'unknown')}, "
            f"seniority_delta_per_year={trajectory.get('delta_per_year', 0):.2f}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            raw = response.choices[0].message.content or ""
            cleaned = _strip_markdown_fences(raw)

            # Try to parse JSON; fallback to mock on failure
            try:
                card = json.loads(cleaned)
                # Ensure required keys exist
                for key in ["summary", "top_strengths", "concern", "interview_probe", "recommendation", "ats_miss"]:
                    if key not in card:
                        card[key] = MOCK_EXPLANATION_CARD[key]
                return card
            except json.JSONDecodeError:
                return MOCK_EXPLANATION_CARD.copy()

        except Exception:
            return MOCK_EXPLANATION_CARD.copy()

    def generate_recruiter_memo(
        self,
        jd_schema: dict,
        top_candidates: list,
        evaluation_results: dict,
    ) -> str:
        """
        Generate a professional recruiter briefing memo for the hiring team.

        Args:
            jd_schema: Extracted JD schema dict.
            top_candidates: List of top candidate dicts (with name, scores, cards).
            evaluation_results: Dict from evaluator.run_evaluation().

        Returns:
            Plain text memo (200-250 words).
        """
        if self._mock_mode:
            return (
                "CANDIDATE SCREENING SUMMARY\n\n"
                "The screening has identified a strong shortlist of candidates for this role. "
                "The top three candidates each demonstrate meaningful technical depth and "
                "relevant project experience aligned with the position requirements.\n\n"
                "A notable pattern across the shortlist is that the highest-ranked candidates "
                "demonstrate consistent career progression rather than simply matching keywords. "
                "Several strong candidates were discovered who used domain-specific terminology "
                "rather than industry buzzwords — a common gap in traditional keyword screening.\n\n"
                "The team should note that the screening is optimized for technical fit and "
                "career trajectory signals. Cultural fit, communication style, and team "
                "dynamics will require direct interview assessment. Treat these rankings as "
                "a starting point for conversation, not a final verdict."
            )

        # Summarize top 3 candidates for the prompt
        top_3_summary = []
        for i, cand in enumerate(top_candidates[:3], 1):
            name = cand.get("name", f"Candidate {i}")
            rec = cand.get("recommendation", "yes")
            summary = cand.get("summary", "")
            top_3_summary.append(f"{i}. {name} ({rec}): {summary[:150]}")

        user_message = (
            f"Role: {jd_schema.get('role_type', 'Unknown role')} | "
            f"Domain: {jd_schema.get('domain', 'Unknown domain')} | "
            f"Seniority: {jd_schema.get('seniority_level', 'mid')}\n\n"
            f"Top 3 candidates:\n" + "\n".join(top_3_summary) + "\n\n"
            f"Evaluation: TalentForge NDCG={evaluation_results.get('pipeline_ndcg', 0):.2f}, "
            f"ATS NDCG={evaluation_results.get('baseline_ndcg', 0):.2f}, "
            f"Improvement={evaluation_results.get('ndcg_improvement_pct', 0):.1f}%"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RECRUITER_MEMO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.6,
                max_tokens=400,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return (
                "CANDIDATE SCREENING SUMMARY\n\n"
                "The screening identified strong candidates for this role. "
                "Please review the explanation cards for detailed assessments."
            )
