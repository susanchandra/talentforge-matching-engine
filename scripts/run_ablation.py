"""
scripts/run_ablation.py — Comprehensive Ablation Study for TalentForge.

Benchmarks 4 system configurations across all 60 synthetic candidates and 3 sample JDs:
1. Keyword ATS Baseline
2. Dense Retrieval Only (BGE Embeddings + FAISS)
3. Hybrid Retrieval (BGE + 4D Structured Scorer + Career Trajectory)
4. Full TalentForge (Hybrid + Cross-Encoder Reranker)

Measures NDCG@10 and Precision@10 against ground-truth labels.
"""

import json
import os
import sys
import time

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.baseline import keyword_ats_rank
from src.embedder import EmbeddingIndex
from src.scorer import StructuredScorer
from src.trajectory import TrajectoryAnalyzer
from src.reranker import CrossEncoderReranker
from src.evaluator import compute_ndcg_at_k, compute_precision_at_k
from src.pipeline import HYBRID_WEIGHTS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CANDIDATES_DIR = os.path.join(DATA_DIR, "synthetic_candidates")
JDS_DIR = os.path.join(DATA_DIR, "sample_jds")
OUTPUT_FILE = os.path.join(DATA_DIR, "cache", "ablation_results.json")


def load_candidates() -> list:
    """Load all 60 synthetic candidate JSON files."""
    candidates = []
    for fname in sorted(os.listdir(CANDIDATES_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(CANDIDATES_DIR, fname), "r", encoding="utf-8") as f:
                candidates.append(json.load(f))
    return candidates


def load_jds() -> dict:
    """Load 3 sample job descriptions."""
    jd_files = {
        "Senior ML Engineer": "jd_ml_engineer_senior.txt",
        "Mid Data Scientist": "jd_data_scientist_mid.txt",
        "MLOps Engineer": "jd_mlops_engineer.txt",
    }
    jds = {}
    for label, fname in jd_files.items():
        with open(os.path.join(JDS_DIR, fname), "r", encoding="utf-8") as f:
            jds[label] = f.read()
    return jds


def build_candidate_schema_mock(cand: dict) -> dict:
    """Heuristic structured schema generation for candidate evaluation."""
    import re
    text = cand.get("resume_text", "")
    exp_match = re.search(r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', text, re.IGNORECASE)
    exp_years = float(exp_match.group(1)) if exp_match else 3.0

    skill_keywords = [
        "Python", "PyTorch", "TensorFlow", "scikit-learn", "Docker", "Kubernetes",
        "MLflow", "Airflow", "SQL", "Spark", "AWS", "GCP", "FastAPI",
        "machine learning", "deep learning", "NLP", "computer vision", "MLOps",
        "LLM", "RAG", "transformer", "statistics", "pandas", "numpy", "Redis",
    ]
    found_skills = [s for s in skill_keywords if s.lower() in text.lower()]

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

    # Extract role history
    role_history = []
    if "intern" in text.lower() or "internship" in text.lower():
        role_history.append({"title": "ML Intern", "seniority_level": "intern", "duration_years": 1.0})
    if exp_years >= 2:
        role_history.append({"title": "Software Engineer", "seniority_level": "mid", "duration_years": min(3.0, exp_years)})
    if exp_years >= 5:
        role_history.append({"title": "Senior ML Engineer", "seniority_level": "senior", "duration_years": exp_years - 3.0})

    if not role_history:
        role_history.append({"title": "Engineer", "seniority_level": seniority, "duration_years": exp_years})

    return {
        "name": cand.get("name", "Unknown"),
        "skills": found_skills or ["Python", "machine learning"],
        "experience_years": exp_years,
        "role_history": role_history,
        "projects": [{"description": text[:300], "technologies": found_skills[:4]}],
        "domain": "machine learning",
    }


def build_jd_schema_mock(jd_text: str) -> dict:
    """Deterministic JD schema extraction for benchmark reproducibility."""
    lower = jd_text.lower()
    seniority = "mid"
    if "5+ years" in lower or "senior" in lower:
        seniority = "senior"
    elif "1-2 years" in lower or "junior" in lower:
        seniority = "junior"

    skills = ["Python", "PyTorch", "Docker", "SQL", "MLflow", "Kubernetes", "Spark", "FastAPI"]
    req = [s for s in skills if s.lower() in lower]

    return {
        "required_skills": req or ["Python", "machine learning"],
        "seniority_level": seniority,
        "role_type": "Machine Learning Engineer",
        "domain": "Machine Learning",
        "key_expectations": ["model development", "production deployment", "pipeline orchestration"],
    }


def run_ablation():
    print("=" * 70)
    print("TALENTFORGE ABLATION BENCHMARK — 60 CANDIDATES x 3 JOB DESCRIPTIONS")
    print("=" * 70)

    start_total = time.time()
    candidates = load_candidates()
    jds = load_jds()
    ground_truth = {c["id"]: c.get("ground_truth_quality", "unknown") for c in candidates}

    print(f"Loaded {len(candidates)} candidates.")
    print(f"Loaded {len(jds)} job descriptions: {list(jds.keys())}")

    # Initialize components
    print("\nInitializing Vector Index & Embedding Model...")
    embedder = EmbeddingIndex()
    for c in candidates:
        embedder.add_candidate(c["id"], c["resume_text"])

    scorer = StructuredScorer()
    trajectory_analyzer = TrajectoryAnalyzer()
    reranker = CrossEncoderReranker()

    # Pre-extract schemas
    cand_schemas = {c["id"]: build_candidate_schema_mock(c) for c in candidates}
    cand_dict = {c["id"]: c for c in candidates}

    results = {}

    for jd_name, jd_text in jds.items():
        print(f"\nEvaluating JD: {jd_name}...")
        jd_schema = build_jd_schema_mock(jd_text)

        # ── 1. Keyword ATS Baseline ──
        ats_input = [{"candidate_id": c["id"], "resume_text": c["resume_text"]} for c in candidates]
        ats_ranked = keyword_ats_rank(jd_text, ats_input)
        ats_ids = [r["candidate_id"] for r in ats_ranked]
        ats_ndcg = compute_ndcg_at_k(ats_ids, ground_truth, k=10)
        ats_p10 = compute_precision_at_k(ats_ids, ground_truth, k=10)

        # ── 2. Dense Retrieval Only (BGE-M3/Small + FAISS) ──
        dense_results = embedder.search(jd_text, top_k=60)
        dense_ids = [r["candidate_id"] for r in dense_results]
        dense_ndcg = compute_ndcg_at_k(dense_ids, ground_truth, k=10)
        dense_p10 = compute_precision_at_k(dense_ids, ground_truth, k=10)

        # ── 3. Hybrid (Dense + 4D Structured Scorer + Trajectory) ──
        dense_lookup = {r["candidate_id"]: r["semantic_score"] for r in dense_results}
        hybrid_scores = []
        for c in candidates:
            cid = c["id"]
            sem_score = dense_lookup.get(cid, 0.0)
            sc = scorer.score_candidate(jd_schema, cand_schemas[cid])
            traj = trajectory_analyzer.analyze(cand_schemas[cid].get("role_history", []))

            hybrid = (
                HYBRID_WEIGHTS["semantic"] * sem_score
                + HYBRID_WEIGHTS["structured"] * sc["structured_score"]
                + HYBRID_WEIGHTS["trajectory"] * traj["trajectory_score"]
                + HYBRID_WEIGHTS["seniority_fit"] * sc["seniority_fit"]
            )
            hybrid_scores.append({
                "candidate_id": cid,
                "hybrid_score": hybrid,
                "resume_text": c["resume_text"],
            })

        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        hybrid_ids = [r["candidate_id"] for r in hybrid_scores]
        hybrid_ndcg = compute_ndcg_at_k(hybrid_ids, ground_truth, k=10)
        hybrid_p10 = compute_precision_at_k(hybrid_ids, ground_truth, k=10)

        # ── 4. Full Pipeline (Hybrid + Cross-Encoder Rerank on top-20) ──
        top20_hybrid = hybrid_scores[:20]
        reranked = reranker.rerank(jd_text, top20_hybrid, top_k=20)
        reranked_ids = [r["candidate_id"] for r in reranked]
        # Append remaining candidates beyond top 20
        full_ids = reranked_ids + hybrid_ids[20:]
        full_ndcg = compute_ndcg_at_k(full_ids, ground_truth, k=10)
        full_p10 = compute_precision_at_k(full_ids, ground_truth, k=10)

        # Track key case studies for Senior ML Engineer
        case_studies = {}
        for target_id, target_label in [
            ("candidate_011", "Arjun Sharma (Hidden Gem)"),
            ("candidate_031", "Ravi Mehta (Keyword Stuffer)"),
            ("candidate_001", "Priya Krishnamurthy (Strong Senior)"),
        ]:
            case_studies[target_label] = {
                "Keyword ATS": ats_ids.index(target_id) + 1 if target_id in ats_ids else None,
                "Dense Only": dense_ids.index(target_id) + 1 if target_id in dense_ids else None,
                "Hybrid": hybrid_ids.index(target_id) + 1 if target_id in hybrid_ids else None,
                "Full TalentForge": full_ids.index(target_id) + 1 if target_id in full_ids else None,
            }

        results[jd_name] = {
            "Keyword ATS": {"NDCG@10": ats_ndcg, "P@10": ats_p10},
            "Dense Only": {"NDCG@10": dense_ndcg, "P@10": dense_p10},
            "Hybrid": {"NDCG@10": hybrid_ndcg, "P@10": hybrid_p10},
            "Full TalentForge": {"NDCG@10": full_ndcg, "P@10": full_p10},
            "case_studies": case_studies,
        }

    # Compute macro-averages
    macro_avg = {}
    for config in ["Keyword ATS", "Dense Only", "Hybrid", "Full TalentForge"]:
        avg_ndcg = sum(results[jd][config]["NDCG@10"] for jd in jds) / len(jds)
        avg_p10 = sum(results[jd][config]["P@10"] for jd in jds) / len(jds)
        macro_avg[config] = {
            "NDCG@10": round(avg_ndcg, 4),
            "P@10": round(avg_p10, 4),
        }

    elapsed = round(time.time() - start_total, 2)

    # Save to JSON
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_time_seconds": elapsed,
        "macro_average": macro_avg,
        "per_jd_results": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved ablation results to: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("MACRO-AVERAGE ABLATION RESULTS (Across 3 JDs, 60 Candidates)")
    print("=" * 70)
    print(f"{'Configuration':<25} | {'NDCG@10':<10} | {'Precision@10':<12}")
    print("-" * 55)
    for config, m in macro_avg.items():
        print(f"{config:<25} | {m['NDCG@10']:<10.4f} | {m['P@10']:<12.4f}")

    print("\n" + "=" * 70)
    print("CASE STUDY: Senior ML Engineer Rank Shifts")
    print("=" * 70)
    cs = results["Senior ML Engineer"]["case_studies"]
    print(f"{'Candidate Profile':<35} | {'ATS':<5} | {'Dense':<6} | {'Hybrid':<7} | {'Full TF':<8}")
    print("-" * 70)
    for label, ranks in cs.items():
        print(f"{label:<35} | #{ranks['Keyword ATS']:<4} | #{ranks['Dense Only']:<5} | #{ranks['Hybrid']:<6} | #{ranks['Full TalentForge']:<7}")
    print("=" * 70)
    print(f"Total benchmark run time: {elapsed} seconds")


if __name__ == "__main__":
    run_ablation()
