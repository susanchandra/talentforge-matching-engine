# TalentForge ⚡
### Explainability-First Hiring Intelligence Pipeline

**Team:** NeuroForge AI  
**Context:** Built for the INDIA RUNS – Data & AI Challenge (Synthetic Benchmark Prototype)

---

> *"Most ATS systems rank candidates by keyword overlap.  
> TalentForge ranks candidates using hybrid semantic retrieval and career-trajectory reasoning,  
> and explains WHY — generating structured recruiter briefing memos."*

---

## Overview

Traditional Applicant Tracking Systems (ATS) count keyword occurrences between job descriptions and resumes. This introduces two systematic failure modes:

1. **False Positives (Keyword Stuffers):** Candidates packing resumes with buzzwords but possessing minimal practical tenure rank at the top.
2. **False Negatives (Hidden Gems):** Strong engineers who describe impactful work in natural, domain-specific language get buried because their terms do not match the exact JD phrasing.

TalentForge addresses this with a two-stage hybrid matching pipeline and evidence-grounded explanation generation.

---

## System Architecture

```
           Job Description (Raw Text)                      Candidate Resume (PDF/Text)
                       │                                                │
                       ▼                                                ▼
         [Pydantic Structured Extraction]                 [Pydantic Structured Extraction]
           OpenAI gpt-4o-mini (JDSchema)                    OpenAI gpt-4o-mini (CandidateSchema)
                       │                                                │
                       ▼                                                ▼
            Target Role & Requirements                        Extracted Experience & Skills
                       │                                                │
                       ├──────────────────────┬─────────────────────────┤
                       │                      │                         │
                       ▼                      ▼                         ▼
            [BGE-M3 Dense Embedding]   [4D Deterministic Scorer]   [Career Trajectory Analyzer]
            1024-dim, L2-normalized    • Skill Token Match (35%)   • Seniority growth velocity
            Exhaustive FAISS IndexFlatIP • Experience Depth (25%)  • Δ seniority / total_years
                       │               • Project Relevance (25%)        │
                       ▼               • Seniority Fit (15%)            │
             Semantic Score (Top-20)          │                         │
                       │                      │                         │
                       └──────────────────────┼─────────────────────────┘
                                              ▼
                                    [Stage 1: Hybrid Blend]
                   hybrid = 0.40×semantic + 0.35×structured + 0.15×trajectory + 0.10×seniority
                                              │
                                              ▼
                                 [Stage 2: Cross-Encoder Rerank]
                            cross-encoder/ms-marco-MiniLM-L-6-v2
                                 (JD[:500], Resume[:1000]) pairs
                                 Min-max normalized logit fusion:
                          final = 0.60×hybrid + 0.40×cross_encoder_norm
                                              │
                                              ▼
                                 [Pydantic Explanation Engine]
                                   OpenAI gpt-4o-mini (T=0.2)
                               • 3 Evidence-based strengths
                               • 1 Specific gap or concern
                               • 1 Targeted interview probe question
                               • Recruiter synthesis memo
```

---

## Technical Specifications

* **Dense Retrieval:** `BAAI/bge-m3` via `FlagEmbedding` (1024-dimensional dense vectors, L2-normalized).
* **Vector Index:** `faiss.IndexFlatIP` ($O(N)$ exact exhaustive inner-product search; mathematically equivalent to cosine similarity after L2 normalization).
* **Deterministic 4D Scorer:** Python-based rule engine evaluating:
  * *Skill Match (35%):* Token overlap and substring containment.
  * *Experience Depth (25%):* Distance penalty between required seniority and candidate years.
  * *Project Relevance (25%):* Token overlap between project descriptions and JD expectations.
  * *Seniority Fit (15%):* Distance between target level and candidate's most recent title.
* **Trajectory Analyzer:** Calculates career progression velocity ($\Delta \text{seniority} / \text{total years}$), normalizing velocity into a 0–1 score.
* **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring truncated pairs (JD 500 chars, Resume 1000 chars) to stay within the 512-token limit.
* **Structured Generation:** OpenAI `gpt-4o-mini` with Pydantic schema enforcement (`client.beta.chat.completions.parse`) at temperature 0.2.
* **UI:** Streamlit interactive application (`app.py`) featuring side-by-side ATS comparison, score breakdown bars, and Plotly career timeline charts.

---

## Benchmark & Dataset Design

The evaluation is conducted on a **synthetic benchmark dataset** of 60 candidate profiles across 6 controlled archetypes:

| Cohort | Count | Profile Description | Expected Behavior |
| :--- | :---: | :--- | :--- |
| **Strong Senior** | 10 | 5–8 yrs experience, production ML systems, clear impact | High rank across ATS and TalentForge |
| **Good Hidden** | 10 | 3–5 yrs, built high-scale systems, uses non-standard terms | **Buried by ATS, elevated by TalentForge** |
| **High Growth** | 10 | 1–3 yrs, rapid title advancement and strong trajectory | Rewarded for velocity over tenure |
| **Keyword Stuffer** | 10 | 1 yr internship, packed with modern buzzwords | **Top 3 on ATS, penalized by TalentForge** |
| **Career Changer** | 10 | Transitioning from adjacent engineering/quant domains | Partial skill transfer credit |
| **Domain Specialist**| 10 | Deep vertical expertise (e.g. CV or NLP only) | Balanced domain match |

*3 Sample JDs:* Senior ML Engineer (`jd_ml_engineer_senior.txt`), Mid Data Scientist (`jd_data_scientist_mid.txt`), MLOps Engineer (`jd_mlops_engineer.txt`).

---

## Quickstart

### 1. Installation
```bash
git clone <repo-url>
cd TalentForge-X
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-openai-key"

# Linux / macOS
export OPENAI_API_KEY="sk-your-openai-key"
```

### 3. Run Benchmark Ablation
```bash
python scripts/run_ablation.py
```

### 4. Launch Streamlit UI
```bash
streamlit run app.py
```

---

## Known Boundaries & Scope

* **Dataset Scope:** Tested exclusively on 60 synthetic profiles; not yet validated on live applicant data.
* **Vector Index:** Flat exhaustive scan ($O(N)$); scaled deployments would require ANN indexing (e.g., HNSW) and streaming index mutations.
* **Cross-Encoder Input Length:** Truncated to 512 tokens (500 chars JD, 1000 chars resume).
* **Latency & Scale:** Offline batch evaluation; sub-millisecond production latency SLAs are unmeasured.
