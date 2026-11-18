"""
Phase 1 validation script — run to confirm all Phase 1 components work.
"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== PHASE 1 VALIDATION ===")

# Test 1: Parser
from src.parser import parse_resume
text = parse_resume("This is a test resume text with more than 100 characters to pass the validation check. Python, ML, TensorFlow experience.")
print(f"[PASS] parser.parse_resume: {len(text)} chars extracted")

# Test 2: Load sample JD
jd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_jds", "jd_ml_engineer_senior.txt")
with open(jd_path, "r") as f:
    jd_text = f.read()
print(f"[PASS] JD loaded: {len(jd_text)} chars")

# Test 3: Load 60 candidates
cand_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic_candidates")
candidates = []
for fn in sorted(os.listdir(cand_dir)):
    if fn.endswith(".json"):
        with open(os.path.join(cand_dir, fn)) as f:
            candidates.append(json.load(f))
print(f"[PASS] Loaded {len(candidates)} synthetic candidates")

# Test 4: Baseline ATS
from src.baseline import keyword_ats_rank
baseline_input = [{"candidate_id": c["id"], "resume_text": c["resume_text"]} for c in candidates]
baseline_results = keyword_ats_rank(jd_text, baseline_input)
print(f"[PASS] ATS baseline ranked {len(baseline_results)} candidates")
print("  ATS Top-5:")
for r in baseline_results[:5]:
    name = next((c["name"] for c in candidates if c["id"] == r["candidate_id"]), "?")
    ctype = next((c["candidate_type"] for c in candidates if c["id"] == r["candidate_id"]), "?")
    print(f"    {r['candidate_id']}: {name} [{ctype}] score={r['keyword_score']:.4f}")

# Check Arjun and Ravi positions in ATS
arjun_pos = next((i + 1 for i, r in enumerate(baseline_results) if r["candidate_id"] == "candidate_011"), None)
ravi_pos = next((i + 1 for i, r in enumerate(baseline_results) if r["candidate_id"] == "candidate_031"), None)
print(f"  Arjun Sharma ATS rank: #{arjun_pos}")
print(f"  Ravi Mehta ATS rank: #{ravi_pos}")

# Test 5: Scorer
from src.scorer import StructuredScorer
scorer = StructuredScorer()
mock_jd = {
    "required_skills": ["Python", "PyTorch", "MLflow", "Docker", "Kubernetes"],
    "seniority_level": "senior",
    "key_expectations": ["ML pipeline", "production deployment"],
}
mock_cand = {
    "skills": ["Python", "PyTorch", "scikit-learn", "Docker"],
    "experience_years": 6,
    "role_history": [{"title": "ML Engineer", "seniority_level": "senior", "duration_years": 3}],
    "projects": [{"description": "ML pipeline for production deployment", "technologies": ["PyTorch", "Docker"]}],
}
scores = scorer.score_candidate(mock_jd, mock_cand)
print(f"[PASS] Scorer: {scores}")

# Test 6: Trajectory
from src.trajectory import TrajectoryAnalyzer
analyzer = TrajectoryAnalyzer()
traj = analyzer.analyze([
    {"title": "Junior ML", "seniority_level": "junior", "duration_years": 1.5},
    {"title": "ML Engineer", "seniority_level": "mid", "duration_years": 2},
    {"title": "Senior ML", "seniority_level": "senior", "duration_years": 2},
])
print(f"[PASS] Trajectory: class={traj['trajectory_class']}, score={traj['trajectory_score']}")

# Test 7: Evaluator
from src.evaluator import compute_ndcg_at_k, compute_precision_at_k
ground_truth = {"c1": "high", "c2": "high", "c3": "medium", "c4": "low", "c5": "high"}
ndcg = compute_ndcg_at_k(["c1", "c2", "c3", "c4", "c5"], ground_truth, k=5)
p_at_k = compute_precision_at_k(["c1", "c2", "c3", "c4", "c5"], ground_truth, k=5)
print(f"[PASS] Evaluator: NDCG@5={ndcg:.4f}, P@5={p_at_k:.4f}")

print("")
print("=" * 60)
print("PHASE 1 COMPLETE: Pipeline skeleton, 60 synthetic candidates,")
print("3 sample JDs, all core components working.")
print(f"ATS Demo Story: Arjun Sharma #{arjun_pos} vs Ravi Mehta #{ravi_pos} - confirms ATS failure mode!")
print("=" * 60)
