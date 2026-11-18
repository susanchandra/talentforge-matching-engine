"""
scripts/precompute_demo.py — Pre-compute all demo results for TalentForge.

Run this ONCE before demo day to generate data/cache/demo_results.json.
After running this script, Demo Mode in the Streamlit app works with
ZERO LLM API calls — it reads only from the cache file.

Usage:
  python scripts/precompute_demo.py

Requires:
  OPENAI_API_KEY environment variable (or will use mock mode)
"""

import json
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import TalentForgePipeline
from src.baseline import keyword_ats_rank
from src.explainer import ExplanationEngine
from src.evaluator import run_evaluation
from src.cache_manager import CacheManager
from src.trajectory import TrajectoryAnalyzer

CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_candidates")
JD_FILE = os.path.join(PROJECT_ROOT, "data", "sample_jds", "jd_ml_engineer_senior.txt")
CACHE_FILE = os.path.join(PROJECT_ROOT, "data", "cache", "demo_results.json")


def load_all_candidates() -> list:
    """Load all 60 synthetic candidate JSON files."""
    candidates = []
    for filename in sorted(os.listdir(CANDIDATES_DIR)):
        if filename.endswith(".json"):
            filepath = os.path.join(CANDIDATES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                candidates.append(json.load(f))
    print(f"Loaded {len(candidates)} candidates.")
    return candidates


def build_candidates_meta(candidates: list) -> dict:
    """Build a candidate_id → metadata mapping."""
    return {
        c["id"]: {
            "ground_truth_quality": c.get("ground_truth_quality", "unknown"),
            "ground_truth_notes": c.get("ground_truth_notes", ""),
            "candidate_type": c.get("candidate_type", "unknown"),
            "name": c.get("name", "Unknown"),
        }
        for c in candidates
    }


def main():
    print("=" * 60)
    print("TalentForge — Pre-computing Demo Results")
    print("=" * 60)

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set — running in mock extraction mode.")
        print("   For best demo results, set OPENAI_API_KEY before running.\n")

    # Load JD
    print(f"Loading JD: {JD_FILE}")
    with open(JD_FILE, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Load candidates
    print(f"Loading candidates from: {CANDIDATES_DIR}")
    candidates = load_all_candidates()

    if len(candidates) < 60:
        print(f"⚠️  Only {len(candidates)} candidates found. Run generate_synthetic_data.py first.")

    # Build metadata mapping
    candidates_meta = build_candidates_meta(candidates)

    # ── Step 1: Initialize pipeline and index candidates ──────────────────────
    print("\n[1/7] Initializing TalentForge pipeline...")
    pipeline = TalentForgePipeline(openai_api_key=api_key if api_key else None)

    print("[2/7] Indexing all candidates (this may take a few minutes)...")
    pipeline.index_candidates(candidates)

    # ── Step 2: Run TalentForge ranking ──────────────────────────────────────
    print("\n[3/7] Running TalentForge hybrid ranking...")
    talentforge_results = pipeline.rank_for_jd(jd_text, top_k=20, use_reranker=True)

    # Print top-10
    print("\n📊 TalentForge Top-10 Results:")
    print(f"  {'Rank':<5} {'Name':<30} {'Final Score':<14} {'Type':<25}")
    print("  " + "-" * 80)
    for result in talentforge_results[:10]:
        ctype = candidates_meta.get(result["candidate_id"], {}).get("candidate_type", "?")
        print(f"  {result['rank']:<5} {result['name']:<30} {result['final_score']:<14.4f} {ctype:<25}")

    # ── Step 3: Run keyword ATS baseline ─────────────────────────────────────
    print("\n[4/7] Running ATS baseline ranking...")
    baseline_input = [
        {"candidate_id": c["id"], "resume_text": c.get("resume_text", "")}
        for c in candidates
    ]
    baseline_results = keyword_ats_rank(jd_text, baseline_input)

    # Print top-10 baseline
    print("\n📊 ATS Baseline Top-10 Results:")
    print(f"  {'Rank':<5} {'Candidate ID':<20} {'ATS Score':<12} {'Type':<25}")
    print("  " + "-" * 70)
    for i, result in enumerate(baseline_results[:10], 1):
        ctype = candidates_meta.get(result["candidate_id"], {}).get("candidate_type", "?")
        name = candidates_meta.get(result["candidate_id"], {}).get("name", "?")
        print(f"  {i:<5} {name:<20} {result['keyword_score']:<12.4f} {ctype:<25}")

    # ── Step 4: Find key candidates for the story ─────────────────────────────
    arjun_tf_rank = next(
        (r["rank"] for r in talentforge_results if "Arjun" in r["name"]), "N/A"
    )
    arjun_ats_rank_pos = next(
        (i + 1 for i, r in enumerate(baseline_results) if "candidate_011" in r["candidate_id"]),
        "N/A"
    )
    ravi_tf_rank = next(
        (r["rank"] for r in talentforge_results if "candidate_031" == r["candidate_id"]), "N/A"
    )
    ravi_ats_rank_pos = next(
        (i + 1 for i, r in enumerate(baseline_results) if "candidate_031" in r["candidate_id"]),
        "N/A"
    )

    print(f"\n✨ Key Story Candidates:")
    print(f"  Arjun Sharma  — ATS rank: {arjun_ats_rank_pos:>3} | TalentForge rank: {arjun_tf_rank}")
    print(f"  Ravi Mehta    — ATS rank: {ravi_ats_rank_pos:>3} | TalentForge rank: {ravi_tf_rank}")

    # ── Step 5: Generate explanation cards for top 10 ────────────────────────
    print("\n[5/7] Generating explanation cards for top 10...")
    explainer = ExplanationEngine(
        openai_client=pipeline.openai_client
    )

    explanation_cards = {}
    trajectory_analyzer = TrajectoryAnalyzer()

    for result in talentforge_results[:10]:
        cid = result["candidate_id"]
        print(f"  Generating card for: {result['name']} (rank {result['rank']})")

        jd_schema = result.get("jd_schema", pipeline._mock_jd_schema(jd_text))
        candidate_schema = result.get("candidate_schema", {})
        scores = {
            "skill_match": result.get("skill_match", 0),
            "experience_depth": result.get("experience_depth", 0),
            "project_relevance": result.get("project_relevance", 0),
            "seniority_fit": result.get("seniority_fit", 0),
        }
        trajectory = result.get("trajectory_data", {})

        card = explainer.generate_explanation_card(
            jd_schema=jd_schema,
            candidate_schema=candidate_schema,
            scores=scores,
            trajectory=trajectory,
            rank=result["rank"],
        )
        explanation_cards[cid] = {
            **card,
            "name": result["name"],
            "rank": result["rank"],
            "final_score": result["final_score"],
        }

    # ── Step 6: Run evaluation ────────────────────────────────────────────────
    print("\n[6/7] Running evaluation...")
    # For evaluation, use top-20 from each system
    tf_top20 = talentforge_results[:20] if len(talentforge_results) >= 20 else talentforge_results
    baseline_top20 = baseline_results[:20]

    evaluation = run_evaluation(
        pipeline_results=tf_top20,
        baseline_results=baseline_top20,
        candidates_meta=candidates_meta,
        k=10,
    )

    print(f"\n📈 Evaluation Results (k=10):")
    print(f"  TalentForge NDCG@10  : {evaluation['pipeline_ndcg']:.4f}")
    print(f"  ATS Baseline NDCG@10 : {evaluation['baseline_ndcg']:.4f}")
    print(f"  NDCG Improvement     : {evaluation['ndcg_improvement_pct']:+.1f}%")
    print(f"  TalentForge P@10     : {evaluation['pipeline_precision']:.4f}")
    print(f"  ATS Baseline P@10    : {evaluation['baseline_precision']:.4f}")

    # ── Step 7: Generate recruiter memo ──────────────────────────────────────
    print("\n[7/7] Generating recruiter memo...")
    top_candidates_for_memo = [
        {
            "name": explanation_cards[r["candidate_id"]]["name"],
            "recommendation": explanation_cards[r["candidate_id"]].get("recommendation", "yes"),
            "summary": explanation_cards[r["candidate_id"]].get("summary", ""),
        }
        for r in talentforge_results[:3]
        if r["candidate_id"] in explanation_cards
    ]

    jd_schema_for_memo = talentforge_results[0].get("jd_schema", pipeline._mock_jd_schema(jd_text)) if talentforge_results else {}
    recruiter_memo = explainer.generate_recruiter_memo(
        jd_schema=jd_schema_for_memo,
        top_candidates=top_candidates_for_memo,
        evaluation_results=evaluation,
    )

    # ── Serialize results to cache ────────────────────────────────────────────
    print("\n💾 Saving results to cache...")

    # Build serializable result dicts (remove resume_text for cache size)
    def clean_result(r: dict) -> dict:
        """Remove large fields for cache."""
        clean = {k: v for k, v in r.items() if k not in ("resume_text",)}
        # Convert nested dicts with candidate_schema minimally
        if "candidate_schema" in clean:
            cs = clean["candidate_schema"]
            clean["candidate_schema"] = {
                "name": cs.get("name", ""),
                "skills": cs.get("skills", []),
                "experience_years": cs.get("experience_years", 0),
                "role_history": cs.get("role_history", []),
                "projects": cs.get("projects", []),
                "education": cs.get("education", {}),
                "domain": cs.get("domain", ""),
            }
        return clean

    cache_data = {
        "jd_text": jd_text,
        "jd_schema": talentforge_results[0].get("jd_schema", {}) if talentforge_results else {},
        "talentforge_ranking": [clean_result(r) for r in talentforge_results],
        "baseline_ranking": baseline_results[:20],
        "explanation_cards": explanation_cards,
        "recruiter_memo": recruiter_memo,
        "evaluation": evaluation,
        "candidates_meta": candidates_meta,
        "arjun_ats_rank": arjun_ats_rank_pos,
        "arjun_tf_rank": arjun_tf_rank,
        "ravi_ats_rank": ravi_ats_rank_pos,
        "ravi_tf_rank": ravi_tf_rank,
    }

    cache_manager = CacheManager()
    cache_manager.save_results(CACHE_FILE, cache_data)

    print("\n" + "=" * 60)
    print("✅ DEMO PRE-COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Cache saved to: {CACHE_FILE}")
    print(f"\nKey metrics:")
    print(f"  TalentForge NDCG@10: {evaluation['pipeline_ndcg']:.4f}")
    print(f"  ATS Baseline NDCG@10: {evaluation['baseline_ndcg']:.4f}")
    print(f"  Improvement: {evaluation['ndcg_improvement_pct']:+.1f}%")
    print(f"  Arjun Sharma — ATS: #{arjun_ats_rank_pos} → TalentForge: #{arjun_tf_rank}")
    print(f"  Ravi Mehta   — ATS: #{ravi_ats_rank_pos} → TalentForge: #{ravi_tf_rank}")


if __name__ == "__main__":
    main()
