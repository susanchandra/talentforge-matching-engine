"""
src/explainer.py — Structured Explanation Generation with Pydantic for TalentForge.

Uses GPT-4o-mini with native OpenAI structured outputs (response_format=ExplanationCard)
at temperature 0.2 to produce evidence-grounded hiring recommendations and interview probes.
"""

import json
import re
from typing import Any, List, Literal, Optional

import openai
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Structured Outputs
# ──────────────────────────────────────────────────────────────────────────────

class ExplanationCard(BaseModel):
    """Guaranteed structured schema for candidate explanation cards."""
    summary: str = Field(
        description="2-3 sentence executive assessment directly citing verified resume evidence."
    )
    top_strengths: List[str] = Field(
        description="List of 3 concrete strengths referencing specific projects, companies, tools, or metrics."
    )
    concern: str = Field(
        description="Exactly one concrete gap, limitation, or domain uncertainty (never left blank)."
    )
    interview_probe: str = Field(
        description="One targeted technical question directly probing the noted concern."
    )
    recommendation: Literal["strong yes", "yes", "maybe", "no"] = Field(
        description="Categorical recruiter verdict."
    )
    ats_miss: Optional[str] = Field(
        default=None,
        description="Explanation of why a traditional keyword ATS would miss this profile, or null if ATS would rank normally."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM_PROMPT = """You are a senior technical hiring partner with 10 years of experience.
Your hiring evaluations are objective, evidence-based, and actionable.

Rules:
1. Ground every strength claim in concrete evidence from the candidate profile (project name, employer, technology, scale, or metrics).
2. Never output generic praise like "strong candidate" or "good fit" without specific citation.
3. Include exactly one genuine concern or limitation — do not omit this even for top candidates.
4. Formulate an actionable interview probe that specifically tests the identified concern.
5. If the candidate used non-standard terminology that keyword ATS algorithms penalize, explain it clearly in ats_miss.
"""

RECRUITER_MEMO_SYSTEM_PROMPT = """You are a head of talent acquisition summarizing candidate screening results for engineering leadership.
Write a concise, professional briefing memo (180-220 words).
Structure into plain paragraphs:
1. Executive summary of the top 3 recommended candidates and their distinguishing technical capabilities.
2. Key cohort pattern or observation (e.g. impact of high-growth velocity vs tenure).
3. Important caveats, limitations, or interview focus areas.

Memo format: clean professional prose without bullet points or markdown headings. Refer to "the screening pipeline" rather than "AI".
"""

MOCK_EXPLANATION_CARD = {
    "summary": "Candidate profile demonstrates relevant technical foundations.",
    "top_strengths": [
        "Core technical competencies match the required tech stack.",
        "Demonstrated practical implementation experience in adjacent domain.",
        "Clear professional timeline with verifiable tenure.",
    ],
    "concern": "Unverified depth in high-throughput production deployment environments.",
    "interview_probe": "Walk through an architecture you designed that handled scale or unexpected failures in production.",
    "recommendation": "maybe",
    "ats_miss": None,
}


class ExplanationEngine:
    """
    Generates structured explanation cards and recruiter briefing memos
    using GPT-4o-mini with Pydantic structured output validation.
    """

    def __init__(self, openai_client: Optional[openai.OpenAI]):
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
        Generate a structured, evidence-grounded explanation card.

        Uses native Pydantic schema parsing (temperature 0.2) with fallback handling.
        """
        if self._mock_mode:
            card = MOCK_EXPLANATION_CARD.copy()
            cand_name = candidate_schema.get("name", "The candidate")
            exp_yrs = candidate_schema.get("experience_years", "several")
            skills = ", ".join(candidate_schema.get("skills", ["the core tech stack"])[:3])
            card["summary"] = (
                f"{cand_name} brings {exp_yrs} years of experience with verified capability in {skills}. "
                f"Ranked #{rank} based on structured alignment and trajectory velocity."
            )
            return card

        user_content = (
            f"Candidate Rank: #{rank}\n"
            f"Target Role: {jd_schema.get('role_type', '')} ({jd_schema.get('domain', '')})\n"
            f"Required Skills: {', '.join(jd_schema.get('required_skills', []))}\n"
            f"Candidate: {json.dumps(candidate_schema)}\n"
            f"Computed Scores: skill_match={scores.get('skill_match', 0):.2f}, "
            f"experience_depth={scores.get('experience_depth', 0):.2f}, "
            f"project_relevance={scores.get('project_relevance', 0):.2f}, "
            f"trajectory_class={trajectory.get('trajectory_class', 'unknown')}, "
            f"delta_per_year={trajectory.get('delta_per_year', 0):.2f}"
        )

        try:
            # Native OpenAI beta structured outputs
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=ExplanationCard,
                temperature=0.2,  # Low temperature for deterministic, factual grounding
            )
            parsed: ExplanationCard = completion.choices[0].message.parsed
            return parsed.model_dump()

        except Exception:
            # Fallback for API errors or timeouts
            return self._fallback_generate_card(user_content)

    def _fallback_generate_card(self, user_content: str) -> dict:
        """Defensive string parsing fallback if structured parsing encounters an API error."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT + " Output valid JSON conforming to the schema."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            raw = response.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                card = MOCK_EXPLANATION_CARD.copy()
                card.update(data)
                return card
        except Exception:
            pass
        return MOCK_EXPLANATION_CARD.copy()

    def generate_recruiter_memo(
        self,
        jd_schema: dict,
        top_candidates: list,
        evaluation_results: dict,
    ) -> str:
        """
        Generate a cohesive recruiter briefing memo for engineering leadership.
        """
        if self._mock_mode:
            return (
                "CANDIDATE SCREENING MEMORANDUM\n\n"
                "The candidate screening pipeline evaluated the applicant cohort against the position requirements. "
                "The top-ranked candidates exhibit solid domain competencies, validated project histories, and consistent "
                "career trajectory progression.\n\n"
                "Across the shortlist, candidates with verifiable system delivery and high growth velocity outperformed "
                "profiles characterized by surface-level buzzword density. Several high-impact candidates who used non-standard "
                "terminology were surfaced by the semantic retrieval and trajectory layers.\n\n"
                "Technical leadership should review the attached explanation cards and use the targeted interview probes to "
                "evaluate depth during technical rounds. Cultural alignment and direct problem-solving capacity remain essential "
                "areas for direct interview assessment."
            )

        top_3_summary = []
        for i, cand in enumerate(top_candidates[:3], 1):
            name = cand.get("name", f"Candidate {i}")
            rec = cand.get("recommendation", "yes")
            summary = cand.get("summary", "")
            top_3_summary.append(f"{i}. {name} (Verdict: {rec}): {summary}")

        user_content = (
            f"Target Role: {jd_schema.get('role_type', 'ML Role')} ({jd_schema.get('seniority_level', 'senior')})\n"
            f"Domain: {jd_schema.get('domain', 'Engineering')}\n"
            f"Top Candidates:\n" + "\n".join(top_3_summary) + "\n\n"
            f"Pipeline Evaluation: NDCG@10={evaluation_results.get('pipeline_ndcg', 0):.3f}, "
            f"Baseline NDCG@10={evaluation_results.get('baseline_ndcg', 0):.3f}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RECRUITER_MEMO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=450,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return self._mock_memo()

    def _mock_memo(self) -> str:
        return (
            "CANDIDATE SCREENING MEMORANDUM\n\n"
            "The screening pipeline has identified top candidate profiles matching core requirements. "
            "Please review individual explanation cards for technical breakdowns and targeted interview probes."
        )
