"""
src/extractor.py — LLM-powered structured extraction with Pydantic for TalentForge.

Uses GPT-4o-mini with native OpenAI structured outputs (response_format=PydanticModel)
to extract guaranteed, validated schemas from JDs and resumes.
"""

import json
import re
from typing import Any, List, Literal, Optional

import openai
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Native Structured Outputs
# ──────────────────────────────────────────────────────────────────────────────

class JDSchema(BaseModel):
    """Structured representation of a Job Description."""
    required_skills: List[str] = Field(
        description="List of required technical skills and tools."
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="List of preferred or bonus technical skills."
    )
    seniority_level: Literal["junior", "mid", "senior", "staff", "principal"] = Field(
        description="Target seniority level for the role."
    )
    role_type: str = Field(
        description="Primary role title or specialization, e.g. 'Machine Learning Engineer'."
    )
    domain: str = Field(
        description="Technical domain, e.g. 'ML Platform', 'Computer Vision', 'Data Science'."
    )
    key_expectations: List[str] = Field(
        default_factory=list,
        description="Top expectations and deliverables for the candidate (max 5 items)."
    )


class RoleHistoryItem(BaseModel):
    """Single past or current employment record."""
    title: str = Field(description="Job title held.")
    company: str = Field(description="Company or employer organization.")
    duration_years: float = Field(description="Duration in this role in years.")
    seniority_level: Literal["intern", "junior", "mid", "senior", "staff", "principal"] = Field(
        description="Evaluated seniority level for this specific title."
    )


class ProjectItem(BaseModel):
    """Technical project entry with descriptions and stack."""
    name: str = Field(description="Name or title of the project.")
    description: str = Field(description="Detailed project summary with metrics and impact.")
    technologies: List[str] = Field(description="List of technologies, libraries, and frameworks used.")


class EducationItem(BaseModel):
    """Academic background details."""
    degree: str = Field(description="Degree obtained or pursued (e.g. B.Tech, M.S., Ph.D.).")
    field: str = Field(description="Field of study (e.g. Computer Science, Statistics).")
    institution: str = Field(description="University, college, or institute name.")


class CandidateSchema(BaseModel):
    """Structured representation of a Candidate Resume."""
    name: str = Field(description="Full name of the candidate.")
    skills: List[str] = Field(description="List of technical and operational skills possessed.")
    experience_years: float = Field(description="Total cumulative years of professional experience.")
    role_history: List[RoleHistoryItem] = Field(
        default_factory=list,
        description="Chronological sequence of roles held."
    )
    projects: List[ProjectItem] = Field(
        default_factory=list,
        description="Notable projects demonstrating domain capability."
    )
    education: EducationItem = Field(
        description="Highest or primary academic credential."
    )
    domain: str = Field(
        description="Primary technical domain specialization."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extraction Functions with Native Structured Outputs
# ──────────────────────────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = (
    "You are an expert technical job description analyst. "
    "Analyze the provided job description and extract a strictly conforming schema. "
    "Accurately identify all required technical tools, minimum seniority, and key responsibilities."
)

CANDIDATE_SYSTEM_PROMPT = (
    "You are a principal technical recruiter parsing a software/ML resume. "
    "Extract a verified structured schema from the resume text. "
    "Accurately calculate experience years and categorize each past role's seniority level."
)


def extract_jd_schema(jd_text: str, client: openai.OpenAI) -> dict:
    """
    Extract structured schema from a job description using native Pydantic output parsing.

    Args:
        jd_text: Raw job description text.
        client: Initialized OpenAI client.

    Returns:
        Validated dictionary conforming to JDSchema.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JD_SYSTEM_PROMPT},
                {"role": "user", "content": jd_text},
            ],
            response_format=JDSchema,
            temperature=0.1,
        )
        parsed: JDSchema = completion.choices[0].message.parsed
        return parsed.model_dump()
    except Exception as e:
        # Fallback to string prompt if parse beta method is unavailable
        return _fallback_extract_jd(jd_text, client, str(e))


def extract_candidate_schema(resume_text: str, client: openai.OpenAI) -> dict:
    """
    Extract structured schema from a candidate resume using native Pydantic output parsing.

    Args:
        resume_text: Raw candidate resume text.
        client: Initialized OpenAI client.

    Returns:
        Validated dictionary conforming to CandidateSchema.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": resume_text},
            ],
            response_format=CandidateSchema,
            temperature=0.1,
        )
        parsed: CandidateSchema = completion.choices[0].message.parsed
        return parsed.model_dump()
    except Exception as e:
        return _fallback_extract_candidate(resume_text, client, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Defensive Fallbacks for Offline / Legacy API Environments
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_extract_jd(jd_text: str, client: openai.OpenAI, error_msg: str) -> dict:
    """Regex-backed JSON parser fallback if native beta.parse fails."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JD_SYSTEM_PROMPT + " Output valid JSON."},
            {"role": "user", "content": jd_text},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or "{}"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Failed to parse JD JSON: {error_msg}")


def _fallback_extract_candidate(resume_text: str, client: openai.OpenAI, error_msg: str) -> dict:
    """Regex-backed JSON parser fallback if native beta.parse fails."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT + " Output valid JSON."},
            {"role": "user", "content": resume_text},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or "{}"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Failed to parse Candidate JSON: {error_msg}")
