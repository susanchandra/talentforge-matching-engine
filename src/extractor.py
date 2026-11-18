"""
src/extractor.py — LLM-powered structured extraction for TalentForge.

Uses GPT-4o-mini to extract structured schemas from JDs and resumes.
"""

import json
import re
from typing import Any

import openai


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output."""
    # Remove leading/trailing whitespace
    text = text.strip()
    # Match optional language tag (json, JSON, etc.)
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _safe_json_parse(raw: str, context: str = "") -> dict:
    """
    Attempt to parse JSON from LLM response with one retry after stripping fences.

    Args:
        raw: Raw string from LLM.
        context: Description for error messages.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If JSON cannot be parsed after retry.
    """
    # First attempt: strip fences then parse
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Second attempt: try to find JSON object within the text
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse JSON from LLM response ({context}). "
        f"Raw response (first 300 chars): {raw[:300]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# JD Schema Extraction
# ──────────────────────────────────────────────────────────────────────────────

JD_SYSTEM_PROMPT = """You are a job description analyst. Extract a structured JSON schema \
from the job description. Return ONLY valid JSON with no preamble, \
no markdown fences, no explanation.
Schema:
{
  "required_skills": ["list of strings"],
  "preferred_skills": ["list of strings"],
  "seniority_level": "one of [junior, mid, senior, staff, principal]",
  "role_type": "string",
  "domain": "string",
  "key_expectations": ["list of strings, max 5"]
}"""


def extract_jd_schema(jd_text: str, client: openai.OpenAI) -> dict:
    """
    Extract structured schema from a job description using GPT-4o-mini.

    Args:
        jd_text: Raw job description text.
        client: Initialized OpenAI client.

    Returns:
        Parsed JD schema dict with keys: required_skills, preferred_skills,
        seniority_level, role_type, domain, key_expectations.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JD_SYSTEM_PROMPT},
            {"role": "user", "content": jd_text},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    raw = response.choices[0].message.content or ""
    return _safe_json_parse(raw, context="JD schema")


# ──────────────────────────────────────────────────────────────────────────────
# Candidate Schema Extraction
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATE_SYSTEM_PROMPT = """You are a technical recruiter parsing a resume. Extract a structured \
JSON schema. Return ONLY valid JSON with no preamble, no markdown fences.
Schema:
{
  "name": "string",
  "skills": ["list of strings"],
  "experience_years": "number",
  "role_history": [
    {
      "title": "string",
      "company": "string",
      "duration_years": "number",
      "seniority_level": "one of [intern, junior, mid, senior, staff, principal]"
    }
  ],
  "projects": [
    {
      "name": "string",
      "description": "string",
      "technologies": ["list of strings"]
    }
  ],
  "education": {
    "degree": "string",
    "field": "string",
    "institution": "string"
  },
  "domain": "string"
}"""


def extract_candidate_schema(resume_text: str, client: openai.OpenAI) -> dict:
    """
    Extract structured schema from a resume using GPT-4o-mini.

    Args:
        resume_text: Raw resume text.
        client: Initialized OpenAI client.

    Returns:
        Parsed candidate schema dict with name, skills, experience_years,
        role_history, projects, education, domain.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": resume_text[:6000]},  # Truncate to avoid token limits
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    raw = response.choices[0].message.content or ""
    return _safe_json_parse(raw, context="candidate schema")
