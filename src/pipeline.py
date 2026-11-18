"""
src/pipeline.py — Master pipeline orchestrator for TalentForge.

Coordinates all components: parsing → extraction → embedding → scoring
→ trajectory analysis → hybrid ranking → cross-encoder reranking.

The pipeline produces a final ranked list with all dimension scores,
trajectory data, and metadata needed by the Streamlit UI and explainer.
"""

import os
from typing import List, Optional

import openai

from src.parser import parse_resume
from src.extractor import extract_jd_schema, extract_candidate_schema
from src.embedder import EmbeddingIndex
from src.scorer import StructuredScorer
from src.trajectory import TrajectoryAnalyzer
from src.reranker import CrossEncoderReranker


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid score weights
# ──────────────────────────────────────────────────────────────────────────────

HYBRID_WEIGHTS = {
    "semantic": 0.40,
    "structured": 0.35,
    "trajectory": 0.15,
    "seniority_fit": 0.10,
}


class TalentForgePipeline:
    """
    Master orchestrator for the TalentForge hybrid ranking system.

    Manages the complete lifecycle from candidate indexing through
    final ranked output with explanation-ready metadata.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the pipeline with all component instances.

        Args:
            openai_api_key: OpenAI API key string. If None, extractor
                            and explainer run in mock mode.
        """
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            # Try environment variable
            env_key = os.environ.get("OPENAI_API_KEY", "")
            self.openai_client = openai.OpenAI(api_key=env_key) if env_key else None

        # Core components
        self.embedder = EmbeddingIndex()
        self.scorer = StructuredScorer()
        self.trajectory_analyzer = TrajectoryAnalyzer()

        # Lazy-loaded (heavy models)
        self._reranker: Optional[CrossEncoderReranker] = None

        # Candidate storage: candidate_id -> {schema, resume_text, metadata}
        self.candidate_store: dict = {}

        print("TalentForgePipeline initialized.")

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Lazy-load the cross-encoder reranker on first use."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    # ── Candidate Indexing ────────────────────────────────────────────────────

    def index_candidates(self, candidates: List[dict]) -> None:
        """
        Parse, extract, and index all candidates into the FAISS vector store.

        Args:
            candidates: List of candidate dicts, each with:
                        - id (str): Unique candidate identifier
                        - resume_text (str): Raw resume text (or PDF path)
                        - Any other metadata fields (preserved in store)
        """
        total = len(candidates)
        print(f"Indexing {total} candidates...")

        for i, candidate in enumerate(candidates, 1):
            cid = candidate.get("id", candidate.get("candidate_id", f"cand_{i}"))

            # Parse resume text
            try:
                resume_text = parse_resume(candidate.get("resume_text", ""))
            except (ValueError, Exception) as e:
                print(f"  [WARN] Skipping {cid}: {e}")
                continue

            # Extract structured schema (use mock if no API key)
            if self.openai_client:
                try:
                    schema = extract_candidate_schema(resume_text, self.openai_client)
                except Exception as e:
                    print(f"  [WARN] Schema extraction failed for {cid}: {e}")
                    schema = self._mock_candidate_schema(candidate)
            else:
                schema = self._mock_candidate_schema(candidate)

            # Ensure name is set
            if not schema.get("name"):
                schema["name"] = candidate.get("name", cid)

            # Add to FAISS index
            self.embedder.add_candidate(cid, resume_text)

            # Store schema and metadata
            self.candidate_store[cid] = {
                "schema": schema,
                "resume_text": resume_text,
                "ground_truth_quality": candidate.get("ground_truth_quality", "unknown"),
                "ground_truth_notes": candidate.get("ground_truth_notes", ""),
                "candidate_type": candidate.get("candidate_type", "unknown"),
                "name": candidate.get("name", schema.get("name", cid)),
            }

            if i % 10 == 0 or i == total:
                print(f"  Indexed {i}/{total} candidates.")

        print(f"Indexing complete. {len(self.candidate_store)} candidates in store.")

    def _mock_candidate_schema(self, candidate: dict) -> dict:
        """
        Generate a minimal candidate schema from raw metadata when no API key is present.
        Parses basic fields from the candidate dict for scoring purposes.
        """
        import re
        resume_text = candidate.get("resume_text", "")
        
        # Extract experience years from resume text using regex
        exp_match = re.search(r'(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience', resume_text, re.IGNORECASE)
        exp_years = float(exp_match.group(1)) if exp_match else 3.0
        
        # Extract some skills (simple word list approach)
        skill_keywords = [
            "Python", "PyTorch", "TensorFlow", "scikit-learn", "Docker", "Kubernetes",
            "MLflow", "Airflow", "SQL", "Spark", "AWS", "GCP", "Azure", "FastAPI",
            "machine learning", "deep learning", "NLP", "computer vision", "MLOps",
            "LLM", "RAG", "transformer", "BERT", "data science", "statistics",
            "pandas", "numpy", "matplotlib", "Kafka", "Redis", "PostgreSQL",
        ]
        found_skills = [s for s in skill_keywords if s.lower() in resume_text.lower()]
        
        # Determine seniority from experience
        if exp_years < 2:
            seniority = "junior"
        elif exp_years < 5:
            seniority = "mid"
        elif exp_years < 9:
            seniority = "senior"
        elif exp_years < 14:
            seniority = "staff"
        else:
            seniority = "principal"
        
        return {
            "name": candidate.get("name", "Unknown"),
            "skills": found_skills or ["machine learning", "Python"],
            "experience_years": exp_years,
            "role_history": [
                {
                    "title": "Software Engineer",
                    "company": "Unknown Company",
                    "duration_years": exp_years,
                    "seniority_level": seniority,
                }
            ],
            "projects": [],
            "education": {
                "degree": "B.Tech",
                "field": "Computer Science",
                "institution": "Unknown University",
            },
            "domain": "machine learning",
        }

    # ── Ranking ───────────────────────────────────────────────────────────────

    def rank_for_jd(
        self,
        jd_text: str,
        top_k: int = 10,
        use_reranker: bool = True,
    ) -> List[dict]:
        """
        Run the full hybrid ranking pipeline for a job description.

        Pipeline stages:
        1. Extract JD schema (LLM)
        2. Semantic retrieval: top-20 via FAISS
        3. Structured scoring (4 dimensions, deterministic)
        4. Trajectory analysis
        5. Compute hybrid score
        6. Cross-encoder reranking (top-20 → top-k)

        Args:
            jd_text: Raw job description text.
            top_k: Number of final candidates to return.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of top_k result dicts with all scores and metadata.
        """
        # Step 1: Extract JD schema
        if self.openai_client:
            try:
                jd_schema = extract_jd_schema(jd_text, self.openai_client)
            except Exception as e:
                print(f"[WARN] JD extraction failed: {e}. Using mock schema.")
                jd_schema = self._mock_jd_schema(jd_text)
        else:
            jd_schema = self._mock_jd_schema(jd_text)

        # Step 2: Semantic retrieval (top 20)
        retrieval_results = self.embedder.search(jd_text, top_k=20)

        if not retrieval_results:
            return []

        # Step 3 & 4: Score and analyze trajectory for each retrieved candidate
        scored_candidates = []

        for retrieval in retrieval_results:
            cid = retrieval["candidate_id"]
            semantic_score = retrieval["semantic_score"]

            stored = self.candidate_store.get(cid, {})
            candidate_schema = stored.get("schema", {})
            resume_text = stored.get("resume_text", "")

            # Structured scoring
            scores = self.scorer.score_candidate(jd_schema, candidate_schema)

            # Trajectory analysis
            role_history = candidate_schema.get("role_history", [])
            trajectory = self.trajectory_analyzer.analyze(role_history)

            # Compute hybrid score
            hybrid_score = (
                HYBRID_WEIGHTS["semantic"] * semantic_score
                + HYBRID_WEIGHTS["structured"] * scores["structured_score"]
                + HYBRID_WEIGHTS["trajectory"] * trajectory["trajectory_score"]
                + HYBRID_WEIGHTS["seniority_fit"] * scores["seniority_fit"]
            )

            scored_candidates.append({
                "candidate_id": cid,
                "name": stored.get("name", candidate_schema.get("name", cid)),
                "semantic_score": round(semantic_score, 4),
                "structured_score": scores["structured_score"],
                "skill_match": scores["skill_match"],
                "experience_depth": scores["experience_depth"],
                "project_relevance": scores["project_relevance"],
                "seniority_fit": scores["seniority_fit"],
                "trajectory_score": trajectory["trajectory_score"],
                "trajectory_class": trajectory["trajectory_class"],
                "hybrid_score": round(hybrid_score, 4),
                "resume_text": resume_text,
                "candidate_schema": candidate_schema,
                "trajectory_data": trajectory,
                "jd_schema": jd_schema,
                "ground_truth_quality": stored.get("ground_truth_quality", "unknown"),
                "candidate_type": stored.get("candidate_type", "unknown"),
            })

        # Sort by hybrid score
        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Step 5: Cross-encoder reranking
        if use_reranker and len(scored_candidates) > 1:
            rerank_input = [
                {
                    "candidate_id": c["candidate_id"],
                    "resume_text": c["resume_text"],
                    "hybrid_score": c["hybrid_score"],
                }
                for c in scored_candidates
            ]
            reranked = self.reranker.rerank(jd_text, rerank_input, top_k=top_k)

            # Merge rerank scores back into full result dicts
            rerank_lookup = {r["candidate_id"]: r for r in reranked}
            final_results = []
            for cid_result in reranked:
                cid = cid_result["candidate_id"]
                full = next(c for c in scored_candidates if c["candidate_id"] == cid)
                full["final_score"] = cid_result["final_score"]
                full["cross_encoder_score"] = cid_result["cross_encoder_score"]
                final_results.append(full)

        else:
            # No reranker: use hybrid score as final
            final_results = scored_candidates[:top_k]
            for r in final_results:
                r["final_score"] = r["hybrid_score"]
                r["cross_encoder_score"] = 0.0

        # Add rank positions
        for i, result in enumerate(final_results, 1):
            result["rank"] = i

        return final_results

    def _mock_jd_schema(self, jd_text: str) -> dict:
        """Generate a minimal JD schema without LLM when API key is unavailable."""
        import re
        
        # Detect seniority from JD text
        jd_lower = jd_text.lower()
        if "senior" in jd_lower or "5+" in jd_lower or "lead" in jd_lower:
            seniority = "senior"
        elif "mid" in jd_lower or "3+" in jd_lower:
            seniority = "mid"
        elif "junior" in jd_lower or "entry" in jd_lower:
            seniority = "junior"
        elif "staff" in jd_lower or "principal" in jd_lower:
            seniority = "staff"
        else:
            seniority = "mid"

        # Extract common ML skills
        skill_pool = [
            "Python", "machine learning", "PyTorch", "TensorFlow", "scikit-learn",
            "MLflow", "Docker", "Kubernetes", "SQL", "deep learning", "NLP",
            "MLOps", "data science", "AWS", "GCP", "Spark", "Airflow",
        ]
        required_skills = [s for s in skill_pool if s.lower() in jd_lower][:8]
        if not required_skills:
            required_skills = ["Python", "machine learning", "data science"]

        return {
            "required_skills": required_skills,
            "preferred_skills": [],
            "seniority_level": seniority,
            "role_type": "ML Engineer",
            "domain": "machine learning",
            "key_expectations": [
                "Build and deploy ML models",
                "Collaborate with cross-functional teams",
                "Drive data-driven decisions",
            ],
        }
