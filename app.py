"""
app.py — TalentForge Streamlit Application
Team: NeuroForge AI | INDIA RUNS – Data & AI Challenge

Explainability-first ML hiring intelligence.
"""

import json
import os
import io
import sys
import time
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TalentForge — Explainability-First Hiring AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialization (all state in one block)
# ──────────────────────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "api_key": "",
        "page": "Home",
        "pipeline": None,
        "run_results": None,
        "baseline_results": None,
        "explanation_cards": None,
        "recruiter_memo": None,
        "evaluation": None,
        "jd_text": "",
        "jd_schema": None,
        "candidates_meta": {},
        "demo_cache": None,
        "run_complete": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_candidates")
SAMPLE_JDS_DIR = os.path.join(PROJECT_ROOT, "data", "sample_jds")
CACHE_FILE = os.path.join(PROJECT_ROOT, "data", "cache", "demo_results.json")

SAMPLE_JD_OPTIONS = {
    "Senior ML Engineer": "jd_ml_engineer_senior.txt",
    "Mid-Level Data Scientist": "jd_data_scientist_mid.txt",
    "MLOps Engineer": "jd_mlops_engineer.txt",
}

RECOMMENDATION_COLORS = {
    "strong yes": "#00c853",
    "yes": "#2196f3",
    "maybe": "#ff9800",
    "no": "#f44336",
}

RECOMMENDATION_ICONS = {
    "strong yes": "✅",
    "yes": "👍",
    "maybe": "🤔",
    "no": "❌",
}

TRAJECTORY_ICONS = {
    "high_growth": "🚀",
    "steady_growth": "📈",
    "lateral": "➡️",
    "declining": "📉",
    "unknown": "❓",
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
<style>
/* ── Import Google Font ─────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────────────── */
.css-1d391kg, section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #1a1a3e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
}

/* ── Hero section ────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d2050 100%);
    padding: 60px 40px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(74, 144, 217, 0.1) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(135deg, #4a90d9 0%, #7b5ea7 50%, #4a90d9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -2px;
    position: relative;
}
.hero-tagline {
    font-size: 22px;
    color: #a0a8c0;
    margin-top: 16px;
    font-weight: 400;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    position: relative;
    line-height: 1.6;
}
.hero-team {
    font-size: 14px;
    color: #4a90d9;
    margin-top: 20px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    position: relative;
}

/* ── Stat cards ──────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #1a1a3e 0%, #0d2050 100%);
    border: 1px solid rgba(74, 144, 217, 0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(74, 144, 217, 0.2);
}
.stat-number {
    font-size: 48px;
    font-weight: 800;
    color: #4a90d9;
    line-height: 1;
}
.stat-label {
    font-size: 14px;
    color: #a0a8c0;
    margin-top: 8px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Explanation cards ───────────────────── */
.exp-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border-left: 5px solid #4a90d9;
    border: 1px solid #eef0f8;
    transition: box-shadow 0.2s ease;
}
.exp-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}
.exp-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.exp-name {
    font-size: 22px;
    font-weight: 700;
    color: #1a1a2e;
}
.exp-rank {
    font-size: 13px;
    color: #666;
    font-weight: 500;
}
.exp-summary {
    color: #444;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 16px;
    border-left: 3px solid #eef0f8;
    padding-left: 14px;
}
.strength-item {
    padding: 8px 14px;
    background: #f0f8f0;
    border-radius: 8px;
    margin: 4px 0;
    font-size: 14px;
    color: #2d5a3d;
    border-left: 3px solid #4caf50;
}
.concern-box {
    padding: 12px 16px;
    background: #fff8f0;
    border-radius: 8px;
    border-left: 3px solid #ff9800;
    margin: 10px 0;
    font-size: 14px;
    color: #8b5a00;
}
.probe-box {
    padding: 12px 16px;
    background: #f0f4ff;
    border-radius: 8px;
    border-left: 3px solid #3f51b5;
    margin: 10px 0;
    font-size: 14px;
    color: #283593;
    font-style: italic;
}
.ats-miss-box {
    padding: 14px 18px;
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-radius: 10px;
    border: 2px solid #f39c12;
    margin: 12px 0;
    font-size: 14px;
    color: #7d5100;
}
.ats-miss-title {
    font-weight: 700;
    color: #c0392b;
    margin-bottom: 4px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.rec-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Score bars ──────────────────────────── */
.score-bar-container {
    margin: 4px 0;
}
.score-label {
    font-size: 12px;
    color: #666;
    font-weight: 500;
    margin-bottom: 2px;
}
.score-bar-bg {
    background: #f0f0f0;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4a90d9, #7b5ea7);
    transition: width 0.5s ease;
}

/* ── Demo banner ─────────────────────────── */
.demo-banner {
    background: linear-gradient(135deg, #1a1a3e, #2d1b4e);
    border: 2px solid #7b5ea7;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Comparison highlight ────────────────── */
.rank-improve {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
.rank-decline {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    border-left: 4px solid #ef5350;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* ── Section headers ─────────────────────── */
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 3px solid #4a90d9;
    display: inline-block;
}
.metric-highlight {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #90caf9;
}
.metric-big {
    font-size: 42px;
    font-weight: 800;
    color: #1565c0;
}
.metric-label {
    font-size: 13px;
    color: #1565c0;
    font-weight: 600;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def load_sample_jd(filename: str) -> str:
    """Load a sample JD from the data directory."""
    path = os.path.join(SAMPLE_JDS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_all_candidates() -> list:
    """Load all synthetic candidate JSON files."""
    candidates = []
    if not os.path.exists(CANDIDATES_DIR):
        return candidates
    for filename in sorted(os.listdir(CANDIDATES_DIR)):
        if filename.endswith(".json"):
            filepath = os.path.join(CANDIDATES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                candidates.append(json.load(f))
    return candidates


def load_demo_cache() -> Optional[dict]:
    """Load pre-computed demo results from cache file."""
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pipeline():
    """Get or create the TalentForge pipeline (cached in session state)."""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing TalentForge pipeline (loading embedding model)..."):
            # Add src to path
            if PROJECT_ROOT not in sys.path:
                sys.path.insert(0, PROJECT_ROOT)
            from src.pipeline import TalentForgePipeline
            api_key = st.session_state.api_key or os.environ.get("OPENAI_API_KEY", "")
            st.session_state.pipeline = TalentForgePipeline(openai_api_key=api_key if api_key else None)
    return st.session_state.pipeline


def recommendation_badge(rec: str) -> str:
    """Generate HTML for a recommendation badge."""
    rec_lower = rec.lower()
    color = RECOMMENDATION_COLORS.get(rec_lower, "#999")
    icon = RECOMMENDATION_ICONS.get(rec_lower, "❓")
    return (
        f'<span class="rec-badge" style="background:{color}; color:white;">'
        f'{icon} {rec.upper()}</span>'
    )


def score_bar_html(label: str, score: float) -> str:
    """Generate HTML for a score progress bar."""
    pct = int(score * 100)
    return f"""
<div class="score-bar-container">
  <div class="score-label">{label}: {pct}%</div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{pct}%;"></div>
  </div>
</div>"""


def render_explanation_card(card: dict, result: dict, rank: int):
    """Render a full explanation card for a candidate."""
    name = card.get("name", result.get("name", "Unknown"))
    rec = card.get("recommendation", "maybe")
    summary = card.get("summary", "")
    strengths = card.get("top_strengths", [])
    concern = card.get("concern", "")
    probe = card.get("interview_probe", "")
    ats_miss = card.get("ats_miss")
    final_score = card.get("final_score", result.get("final_score", 0))
    traj_class = result.get("trajectory_class", "unknown")
    traj_icon = TRAJECTORY_ICONS.get(traj_class, "❓")

    st.markdown(f"""
<div class="exp-card">
  <div class="exp-card-header">
    <div>
      <div class="exp-name">#{rank} {name}</div>
      <div class="exp-rank">Final Score: {final_score:.3f} | Trajectory: {traj_icon} {traj_class}</div>
    </div>
    <div>{recommendation_badge(rec)}</div>
  </div>
  <div class="exp-summary">{summary}</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**💪 Top Strengths**")
        for strength in strengths:
            st.markdown(f'<div class="strength-item">✓ {strength}</div>', unsafe_allow_html=True)

    with col2:
        skill_score = result.get("skill_match", 0)
        exp_score = result.get("experience_depth", 0)
        proj_score = result.get("project_relevance", 0)
        sen_score = result.get("seniority_fit", 0)

        st.markdown(f"**📊 Score Breakdown**")
        st.markdown(score_bar_html("Skill Match", skill_score), unsafe_allow_html=True)
        st.markdown(score_bar_html("Experience Depth", exp_score), unsafe_allow_html=True)
        st.markdown(score_bar_html("Project Relevance", proj_score), unsafe_allow_html=True)
        st.markdown(score_bar_html("Seniority Fit", sen_score), unsafe_allow_html=True)

    if concern:
        st.markdown(f'<div class="concern-box">⚠️ <strong>Gap to Probe:</strong> {concern}</div>',
                    unsafe_allow_html=True)

    if probe:
        st.markdown(f'<div class="probe-box">💬 <strong>Interview Question:</strong> {probe}</div>',
                    unsafe_allow_html=True)

    if ats_miss:
        st.markdown(f"""
<div class="ats-miss-box">
  <div class="ats-miss-title">🔍 ATS Would Have Missed This Candidate</div>
  {ats_miss}
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
  <div style="font-size:32px; font-weight:800; color:#4a90d9; letter-spacing:-1px;">
    ⚡ TalentForge
  </div>
  <div style="font-size:13px; color:#7b85a0; margin-top:6px; letter-spacing:2px;">
    NEUROFORGE AI
  </div>
  <div style="margin-top:8px; padding:4px 12px; background:rgba(74,144,217,0.15);
              border-radius:20px; display:inline-block; font-size:11px; color:#4a90d9;">
    INDIA RUNS Challenge
  </div>
</div>
<hr style="border-color:rgba(255,255,255,0.1); margin:16px 0;">
""", unsafe_allow_html=True)

        pages = ["🏠 Home", "🔍 Run Analysis", "⚡ Demo Mode"]
        selection = st.radio("Navigate", pages, label_visibility="collapsed")
        page_map = {
            "🏠 Home": "Home",
            "🔍 Run Analysis": "Run Analysis",
            "⚡ Demo Mode": "Demo Mode",
        }
        st.session_state.page = page_map[selection]

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px; color:#7b85a0; margin-bottom:8px;'>🔑 OpenAI API Key</div>",
                    unsafe_allow_html=True)
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.markdown(
                "<div style='font-size:11px; color:#4caf50;'>✓ API key set</div>",
                unsafe_allow_html=True
            )
        else:
            env_key = os.environ.get("OPENAI_API_KEY", "")
            if env_key:
                st.markdown(
                    "<div style='font-size:11px; color:#4caf50;'>✓ Using OPENAI_API_KEY env var</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='font-size:11px; color:#ff9800;'>⚠ No key — mock mode active</div>",
                    unsafe_allow_html=True
                )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:11px; color:#5a6080; padding-bottom:10px; line-height:1.8;">
  <div>📐 <strong style="color:#7b85a0;">Architecture</strong></div>
  <div>BGE-M3 Embeddings</div>
  <div>FAISS Vector Search</div>
  <div>4D Structured Scoring</div>
  <div>Career Trajectory AI</div>
  <div>Cross-Encoder Reranking</div>
  <div>GPT-4o-mini Explainer</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: Home
# ──────────────────────────────────────────────────────────────────────────────

def render_home():
    inject_css()

    # Hero section
    st.markdown("""
<div class="hero-container">
  <div class="hero-title">TalentForge</div>
  <div class="hero-tagline">
    "Most ATS systems rank candidates by keyword overlap.<br>
    TalentForge ranks candidates using recruiter-style reasoning<br>
    and explains <em>why</em> — making AI hiring trustworthy."
  </div>
  <div class="hero-team">NeuroForge AI · INDIA RUNS – Data & AI Challenge</div>
</div>
""", unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
<div class="stat-card">
  <div class="stat-number">60+</div>
  <div class="stat-label">Candidates Indexed</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="stat-card">
  <div class="stat-number">4</div>
  <div class="stat-label">Scoring Dimensions</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="stat-card">
  <div class="stat-number">5-Stage</div>
  <div class="stat-label">Hybrid Pipeline</div>
</div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
<div class="stat-card">
  <div class="stat-number">100%</div>
  <div class="stat-label">Explainability-First</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture overview
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">How TalentForge Works</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        stages = [
            ("🔍", "Semantic Retrieval", "BGE-M3 embeddings + FAISS index finds semantically similar candidates. Goes beyond keyword matching to understand meaning and context."),
            ("📐", "4D Structured Scoring", "Skill match, experience depth, project relevance, and seniority fit — computed without LLM calls for speed and consistency."),
            ("📈", "Career Trajectory", "Analyzes seniority progression velocity. Rewards candidates who are growing into the role, not just currently at the level."),
            ("🎯", "Cross-Encoder Reranking", "ms-marco cross-encoder scores each (JD, resume) pair jointly for fine-grained relevance — the most accurate but expensive step."),
            ("💡", "Explanation Generation", "GPT-4o-mini writes evidence-based recruiter cards with specific project citations, gaps, and targeted interview questions."),
        ]

        for icon, title, desc in stages:
            st.markdown(f"""
<div style="display:flex; gap:16px; margin-bottom:20px; align-items:flex-start;">
  <div style="font-size:28px; min-width:40px;">{icon}</div>
  <div>
    <div style="font-weight:700; font-size:16px; color:#1a1a2e; margin-bottom:4px;">{title}</div>
    <div style="color:#666; font-size:14px; line-height:1.6;">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Why This Matters</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
<div style="background:linear-gradient(135deg,#fff8f0,#fff3e0); border:2px solid #ff9800;
            border-radius:16px; padding:24px; margin-bottom:16px;">
  <div style="font-weight:700; font-size:16px; color:#e65100; margin-bottom:12px;">
    ❌ The ATS Problem
  </div>
  <div style="color:#bf360c; font-size:14px; line-height:1.8;">
    <strong>Ravi Mehta</strong> — 1-year intern with a buzzword-packed resume —
    ranks <strong>#2 on traditional ATS</strong>. His resume contains every keyword:
    MLOps, LLM, RAG, Docker, Kubernetes, PyTorch...<br><br>
    But he has zero real impact. No shipped products. No users served.
  </div>
</div>

<div style="background:linear-gradient(135deg,#f0fff4,#e8f5e9); border:2px solid #4caf50;
            border-radius:16px; padding:24px;">
  <div style="font-weight:700; font-size:16px; color:#1b5e20; margin-bottom:12px;">
    ✅ The TalentForge Fix
  </div>
  <div style="color:#2e7d32; font-size:14px; line-height:1.8;">
    <strong>Arjun Sharma</strong> — who led ML infrastructure serving <strong>2M farmers</strong>
    — ranks <strong>#30 on ATS</strong>. He uses "ML infrastructure" not "MLOps",
    "model training pipeline" not "CI/CD for models".<br><br>
    TalentForge finds him and ranks him <strong>#2</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Differentiators
    st.markdown('<div class="section-header">What Makes TalentForge Different</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    diffs = [
        ("🔬", "Evidence-Based Explanations", "Every ranking decision is backed by specific evidence from the resume — project names, impact numbers, company names. No generic 'good fit' summaries."),
        ("⚠️", "Mandatory Concern Field", "Every explanation card includes exactly one gap or uncertainty — even for top candidates. This makes the AI trustworthy and the hiring decision more informed."),
        ("📊", "Baseline Comparison", "TalentForge shows you what a keyword ATS would have ranked — and proves why its approach finds better candidates using NDCG and Precision metrics."),
    ]
    for col, (icon, title, desc) in zip([d1, d2, d3], diffs):
        with col:
            st.markdown(f"""
<div style="background:white; border-radius:16px; padding:24px; height:200px;
            box-shadow:0 4px 24px rgba(0,0,0,0.07); border-top:4px solid #4a90d9;">
  <div style="font-size:32px; margin-bottom:12px;">{icon}</div>
  <div style="font-weight:700; font-size:15px; color:#1a1a2e; margin-bottom:8px;">{title}</div>
  <div style="color:#666; font-size:13px; line-height:1.6;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 Use **Run Analysis** to analyze your own JD and resumes, or try **Demo Mode** for an instant preview with pre-computed results.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: Run Analysis
# ──────────────────────────────────────────────────────────────────────────────

def render_run_analysis():
    inject_css()
    st.markdown('<h1 style="color:#1a1a2e; font-weight:800;">🔍 Run Analysis</h1>', unsafe_allow_html=True)

    # ── Section A: Input ──────────────────────────────────────────────────────
    with st.expander("📋 Job Description & Candidate Input", expanded=not st.session_state.run_complete):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Job Description**")
            jd_source = st.radio("JD Source", ["Type / Paste", "Load Sample JD"], horizontal=True, label_visibility="collapsed")

            if jd_source == "Load Sample JD":
                selected_jd = st.selectbox("Select Sample JD", list(SAMPLE_JD_OPTIONS.keys()))
                jd_text = load_sample_jd(SAMPLE_JD_OPTIONS[selected_jd])
                st.text_area("Job Description Preview", jd_text, height=250, disabled=True)
            else:
                jd_text = st.text_area(
                    "Paste your job description here",
                    height=250,
                    placeholder="Senior Machine Learning Engineer\n\nWe are looking for...",
                    label_visibility="collapsed",
                )

        with col2:
            st.markdown("**Candidate Source**")
            use_synthetic = st.checkbox("Use all 60 synthetic candidates", value=True)

            if not use_synthetic:
                uploaded_files = st.file_uploader(
                    "Upload PDF resumes",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help="Upload PDF resume files",
                )
                if uploaded_files:
                    st.success(f"{len(uploaded_files)} file(s) uploaded")

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_button = st.button(
                "⚡ Run TalentForge",
                type="primary",
                use_container_width=True,
                disabled=not jd_text.strip(),
            )
        with col_btn2:
            if not jd_text.strip():
                st.warning("Please provide a job description to run analysis.")

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    if run_button and jd_text.strip():
        try:
            # Add project to path
            if PROJECT_ROOT not in sys.path:
                sys.path.insert(0, PROJECT_ROOT)
            from src.pipeline import TalentForgePipeline
            from src.baseline import keyword_ats_rank
            from src.explainer import ExplanationEngine
            from src.evaluator import run_evaluation

            st.session_state.jd_text = jd_text

            pipeline = get_pipeline()

            # Load candidates
            with st.spinner("Loading candidates..."):
                if use_synthetic:
                    candidates = load_all_candidates()
                    if not candidates:
                        st.error("No synthetic candidates found. Run `python scripts/generate_synthetic_data.py` first.")
                        return
                else:
                    candidates = []  # Would process uploaded PDFs
                    st.warning("PDF upload mode: using synthetic dataset for now.")
                    candidates = load_all_candidates()

            # Index candidates
            with st.spinner(f"Indexing {len(candidates)} candidates..."):
                pipeline.index_candidates(candidates)

            # Run TalentForge ranking
            with st.spinner("Running hybrid ranking pipeline..."):
                results = pipeline.rank_for_jd(jd_text, top_k=20, use_reranker=True)
                st.session_state.run_results = results

            # Run ATS baseline
            with st.spinner("Running ATS baseline comparison..."):
                baseline_input = [{"candidate_id": c["id"], "resume_text": c.get("resume_text", "")} for c in candidates]
                baseline_results = keyword_ats_rank(jd_text, baseline_input)
                st.session_state.baseline_results = baseline_results

            # Build meta
            candidates_meta = {
                c["id"]: {
                    "ground_truth_quality": c.get("ground_truth_quality", "unknown"),
                    "candidate_type": c.get("candidate_type", "unknown"),
                    "name": c.get("name", "Unknown"),
                }
                for c in candidates
            }
            st.session_state.candidates_meta = candidates_meta

            # Generate explanations
            explainer = ExplanationEngine(pipeline.openai_client)
            cards = {}
            prog = st.progress(0, text="Generating explanation cards...")
            for i, result in enumerate(results[:10]):
                prog.progress((i + 1) / 10, text=f"Generating card for {result['name']}...")
                card = explainer.generate_explanation_card(
                    jd_schema=result.get("jd_schema", {}),
                    candidate_schema=result.get("candidate_schema", {}),
                    scores={
                        "skill_match": result.get("skill_match", 0),
                        "experience_depth": result.get("experience_depth", 0),
                        "project_relevance": result.get("project_relevance", 0),
                        "seniority_fit": result.get("seniority_fit", 0),
                    },
                    trajectory=result.get("trajectory_data", {}),
                    rank=result["rank"],
                )
                cards[result["candidate_id"]] = {**card, "name": result["name"], "rank": result["rank"], "final_score": result["final_score"]}
            prog.empty()
            st.session_state.explanation_cards = cards

            # Run evaluation
            evaluation = run_evaluation(results, baseline_results, candidates_meta, k=10)
            st.session_state.evaluation = evaluation

            # Generate recruiter memo
            top3 = [{"name": cards[r["candidate_id"]]["name"],
                      "recommendation": cards[r["candidate_id"]].get("recommendation", "yes"),
                      "summary": cards[r["candidate_id"]].get("summary", "")}
                     for r in results[:3] if r["candidate_id"] in cards]
            jd_schema = results[0].get("jd_schema", {}) if results else {}
            st.session_state.jd_schema = jd_schema
            memo = explainer.generate_recruiter_memo(jd_schema, top3, evaluation)
            st.session_state.recruiter_memo = memo

            st.session_state.run_complete = True
            st.success("✅ Analysis complete!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # ── Section B: Results ────────────────────────────────────────────────────
    if st.session_state.run_complete and st.session_state.run_results:
        render_results_tabs(
            results=st.session_state.run_results,
            baseline_results=st.session_state.baseline_results,
            explanation_cards=st.session_state.explanation_cards,
            recruiter_memo=st.session_state.recruiter_memo,
            evaluation=st.session_state.evaluation,
            candidates_meta=st.session_state.candidates_meta,
            jd_schema=st.session_state.jd_schema or {},
        )

        # Download CSV
        st.markdown("---")
        _render_download_csv(st.session_state.run_results)


def _render_download_csv(results: list):
    """Generate and provide CSV download of ranked results."""
    rows = []
    for r in results:
        rows.append({
            "Rank": r.get("rank", ""),
            "Name": r.get("name", ""),
            "Final Score": r.get("final_score", 0),
            "Skill Match": r.get("skill_match", 0),
            "Experience Depth": r.get("experience_depth", 0),
            "Project Relevance": r.get("project_relevance", 0),
            "Seniority Fit": r.get("seniority_fit", 0),
            "Trajectory": r.get("trajectory_class", ""),
            "Semantic Score": r.get("semantic_score", 0),
            "Candidate Type": r.get("candidate_type", ""),
        })
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Ranked Output (CSV)",
        data=csv_bytes,
        file_name="talentforge_ranking.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared Results Tabs
# ──────────────────────────────────────────────────────────────────────────────

def render_results_tabs(
    results: list,
    baseline_results: list,
    explanation_cards: dict,
    recruiter_memo: str,
    evaluation: dict,
    candidates_meta: dict,
    jd_schema: dict,
):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 TalentForge Ranking",
        "⚖️ vs ATS Baseline",
        "💡 Explanation Cards",
        "📈 Career Timelines",
        "📝 Recruiter Memo",
        "📊 Evaluation",
    ])

    # ── Tab 1: Rankings table ─────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">TalentForge Ranking</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        table_data = []
        for r in results:
            traj_icon = TRAJECTORY_ICONS.get(r.get("trajectory_class", "unknown"), "❓")
            table_data.append({
                "Rank": r.get("rank", ""),
                "Name": r.get("name", "Unknown"),
                "Final Score": r.get("final_score", 0),
                "Skill Match": r.get("skill_match", 0),
                "Experience": r.get("experience_depth", 0),
                "Projects": r.get("project_relevance", 0),
                "Seniority": r.get("seniority_fit", 0),
                "Trajectory": f"{traj_icon} {r.get('trajectory_class', '').replace('_', ' ').title()}",
                "Type": r.get("candidate_type", "").replace("_", " ").title(),
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=1, format="%.3f"),
                "Skill Match": st.column_config.ProgressColumn("Skill Match", min_value=0, max_value=1, format="%.2f"),
                "Experience": st.column_config.ProgressColumn("Experience", min_value=0, max_value=1, format="%.2f"),
                "Projects": st.column_config.ProgressColumn("Projects", min_value=0, max_value=1, format="%.2f"),
                "Seniority": st.column_config.ProgressColumn("Seniority", min_value=0, max_value=1, format="%.2f"),
                "Name": st.column_config.TextColumn("Candidate", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="medium"),
            },
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        for r in results[:10]:
            with st.expander(f"#{r.get('rank')} {r.get('name', 'Unknown')} — Click for details"):
                cid = r["candidate_id"]
                if explanation_cards and cid in explanation_cards:
                    card = explanation_cards[cid]
                    st.markdown(f"**Summary:** {card.get('summary', '')}")
                    rec = card.get("recommendation", "maybe")
                    st.markdown(f"**Recommendation:** {recommendation_badge(rec)}", unsafe_allow_html=True)
                else:
                    st.json({k: v for k, v in r.items() if k not in ("resume_text", "candidate_schema", "jd_schema", "trajectory_data")})

    # ── Tab 2: vs ATS Baseline ────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">TalentForge vs ATS Baseline</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Build rank maps
        tf_rank_map = {r["candidate_id"]: r["rank"] for r in results}
        baseline_rank_map = {r["candidate_id"]: i + 1 for i, r in enumerate(baseline_results)}
        name_map = {r["candidate_id"]: r["name"] for r in results}
        # Add names from meta
        for cid, meta in candidates_meta.items():
            if cid not in name_map:
                name_map[cid] = meta.get("name", cid)

        # Highlight interesting cases
        st.markdown("#### 🎯 The Key Comparison Moments")
        col_a, col_b = st.columns(2)

        with col_a:
            # Arjun Sharma (candidate_011)
            arjun_id = "candidate_011"
            arjun_tf = tf_rank_map.get(arjun_id, "N/A (outside top 20)")
            arjun_ats = baseline_rank_map.get(arjun_id, "N/A")
            st.markdown(f"""
<div class="rank-improve">
  <strong>✨ Arjun Sharma — The Hidden Gem</strong><br>
  <span style="color:#1b5e20;">ATS Rank: #{arjun_ats} → TalentForge Rank: #{arjun_tf}</span><br>
  <small>Led ML infrastructure serving 2M users — but uses non-standard terminology.
  ATS penalizes this. TalentForge understands the impact.</small>
</div>""", unsafe_allow_html=True)

        with col_b:
            # Ravi Mehta (candidate_031)
            ravi_id = "candidate_031"
            ravi_tf = tf_rank_map.get(ravi_id, "N/A (outside top 20)")
            ravi_ats = baseline_rank_map.get(ravi_id, "N/A")
            st.markdown(f"""
<div class="rank-decline">
  <strong>⚠️ Ravi Mehta — The Keyword Stuffer</strong><br>
  <span style="color:#b71c1c;">ATS Rank: #{ravi_ats} → TalentForge Rank: #{ravi_tf}</span><br>
  <small>Dense resume with every ML buzzword but only a 1-year internship.
  ATS rewards the keywords. TalentForge sees through it.</small>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>")

        # Side-by-side comparison table (top 15)
        st.markdown("#### Top 15 Side-by-Side Comparison")
        compare_data = []
        all_ids = list(dict.fromkeys([r["candidate_id"] for r in results[:15]] + [r["candidate_id"] for r in baseline_results[:15]]))

        for cid in all_ids[:20]:
            tf_rank = tf_rank_map.get(cid, 99)
            ats_rank = baseline_rank_map.get(cid, 99)
            name = name_map.get(cid, cid)
            ctype = candidates_meta.get(cid, {}).get("candidate_type", "?").replace("_", " ").title()
            quality = candidates_meta.get(cid, {}).get("ground_truth_quality", "?")
            delta = ats_rank - tf_rank  # positive = TalentForge ranks higher (better)
            compare_data.append({
                "Name": name,
                "TF Rank": tf_rank if tf_rank < 90 else "—",
                "ATS Rank": ats_rank if ats_rank < 90 else "—",
                "Δ Rank": delta,
                "Type": ctype,
                "True Quality": quality.title(),
            })

        compare_df = pd.DataFrame(compare_data)
        st.dataframe(
            compare_df,
            use_container_width=True,
            column_config={
                "TF Rank": st.column_config.NumberColumn("TalentForge #", format="%d"),
                "ATS Rank": st.column_config.NumberColumn("ATS #", format="%d"),
                "Δ Rank": st.column_config.NumberColumn("Δ (positive = TF ranks higher)", format="%+d"),
                "True Quality": st.column_config.TextColumn("True Quality"),
            },
            hide_index=True,
        )

    # ── Tab 3: Explanation Cards ──────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Explanation Cards — Top 10</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not explanation_cards:
            st.info("Explanation cards not yet generated. Run analysis first.")
        else:
            for result in results[:10]:
                cid = result["candidate_id"]
                if cid in explanation_cards:
                    card = explanation_cards[cid]
                    render_explanation_card(card, result, result.get("rank", 0))
                    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tab 4: Career Timelines ───────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Career Trajectory Charts — Top 5</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from src.trajectory import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer()
        jd_seniority = jd_schema.get("seniority_level", "senior")

        for result in results[:5]:
            trajectory_data = result.get("trajectory_data", {})
            name = result.get("name", "Unknown")

            if trajectory_data and trajectory_data.get("role_sequence"):
                fig = analyzer.build_timeline_chart(trajectory_data, name, jd_seniority)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No trajectory data available for {name}.")
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab 5: Recruiter Memo ─────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Recruiter Briefing Memo</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if recruiter_memo:
            st.markdown(f"""
<div style="background:white; border-radius:16px; padding:32px; box-shadow:0 4px 24px rgba(0,0,0,0.08);
            border-left:5px solid #4a90d9; max-width:800px;">
  <div style="font-size:13px; color:#999; text-transform:uppercase; letter-spacing:2px;
              margin-bottom:16px;">Candidate Screening Summary</div>
  <div style="font-size:15px; color:#333; line-height:1.9; white-space:pre-wrap;">{recruiter_memo}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>")
            st.download_button(
                label="📥 Download Memo (.txt)",
                data=recruiter_memo.encode("utf-8"),
                file_name="talentforge_recruiter_memo.txt",
                mime="text/plain",
            )
        else:
            st.info("Recruiter memo not yet generated.")

    # ── Tab 6: Evaluation ─────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if evaluation:
            # Key metric highlights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
<div class="metric-highlight">
  <div class="metric-big">{evaluation['pipeline_ndcg']:.3f}</div>
  <div class="metric-label">TalentForge NDCG@10</div>
</div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
<div class="metric-highlight" style="background:linear-gradient(135deg,#fce4ec,#ffcdd2);border-color:#ef9a9a;">
  <div class="metric-big" style="color:#c62828;">{evaluation['baseline_ndcg']:.3f}</div>
  <div class="metric-label" style="color:#c62828;">ATS Baseline NDCG@10</div>
</div>""", unsafe_allow_html=True)
            with col3:
                improvement = evaluation['ndcg_improvement_pct']
                color = "#1b5e20" if improvement > 0 else "#c62828"
                st.markdown(f"""
<div class="metric-highlight" style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-color:#a5d6a7;">
  <div class="metric-big" style="color:{color};">{improvement:+.1f}%</div>
  <div class="metric-label" style="color:{color};">NDCG Improvement</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>")

            # Bar chart comparison
            fig = go.Figure()
            metrics = ["NDCG@10", "Precision@10"]
            tf_vals = [evaluation["pipeline_ndcg"], evaluation["pipeline_precision"]]
            ats_vals = [evaluation["baseline_ndcg"], evaluation["baseline_precision"]]

            fig.add_trace(go.Bar(name="TalentForge", x=metrics, y=tf_vals,
                                  marker_color="#4a90d9", text=[f"{v:.3f}" for v in tf_vals], textposition="outside"))
            fig.add_trace(go.Bar(name="ATS Baseline", x=metrics, y=ats_vals,
                                  marker_color="#ef5350", text=[f"{v:.3f}" for v in ats_vals], textposition="outside"))
            fig.update_layout(
                title="TalentForge vs ATS Baseline",
                barmode="group",
                yaxis_range=[0, 1.1],
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=380,
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
<div style="background:#f8f9ff; border-radius:12px; padding:20px; margin-top:16px;">
  <strong>What these metrics mean:</strong><br><br>
  <strong>NDCG@10 (Normalized Discounted Cumulative Gain)</strong> — Measures ranking quality,
  giving more credit for placing high-quality candidates at the top. A perfect score of 1.0
  means all top-quality candidates are at positions 1-10 in the correct order.<br><br>
  <strong>Precision@10</strong> — The fraction of the top-10 results that are genuinely high-quality
  candidates (ground truth: "high"). A score of 0.7 means 7 of the top 10 are excellent candidates.
</div>""", unsafe_allow_html=True)
        else:
            st.info("Evaluation results not yet computed.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: Demo Mode
# ──────────────────────────────────────────────────────────────────────────────

def render_demo_mode():
    inject_css()

    # Demo banner
    st.markdown("""
<div class="demo-banner">
  <div style="font-size:32px;">⚡</div>
  <div>
    <div style="font-weight:800; font-size:18px; color:#7b5ea7;">DEMO MODE</div>
    <div style="font-size:13px; color:#a0a8c0;">Pre-computed results — zero API calls required</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Load cache
    if st.session_state.demo_cache is None:
        cache = load_demo_cache()
        if cache is None:
            st.error("""
⚠️ Demo cache not found.

Run this command to generate it:
```bash
python scripts/precompute_demo.py
```

Or set your OpenAI API key and use Run Analysis instead.
""")
            st.markdown("### Quick Setup for Demo Mode")
            st.code("# 1. Generate synthetic data\npython scripts/generate_synthetic_data.py\n\n# 2. Pre-compute demo results (requires OpenAI API key)\npython scripts/precompute_demo.py")
            return
        st.session_state.demo_cache = cache

    cache = st.session_state.demo_cache

    # Extract data from cache
    results = cache.get("talentforge_ranking", [])
    baseline_results = cache.get("baseline_ranking", [])
    explanation_cards = cache.get("explanation_cards", {})
    recruiter_memo = cache.get("recruiter_memo", "")
    evaluation = cache.get("evaluation", {})
    candidates_meta = cache.get("candidates_meta", {})
    jd_schema = cache.get("jd_schema", {})
    jd_text = cache.get("jd_text", "")

    arjun_ats = cache.get("arjun_ats_rank", "?")
    arjun_tf = cache.get("arjun_tf_rank", "?")
    ravi_ats = cache.get("ravi_ats_rank", "?")
    ravi_tf = cache.get("ravi_tf_rank", "?")

    # Story callout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div class="rank-improve" style="padding:20px;">
  <div style="font-weight:700; font-size:16px; color:#1b5e20; margin-bottom:8px;">
    ✨ The Hidden Gem: Arjun Sharma
  </div>
  <div style="font-size:28px; font-weight:800; color:#2e7d32;">
    ATS: #{arjun_ats} → TalentForge: #{arjun_tf}
  </div>
  <div style="font-size:13px; color:#388e3c; margin-top:8px;">
    Led ML infrastructure serving 2M farmers at a 15-person startup.
    Uses "model training pipeline" instead of "CI/CD for models".
    ATS never found him.
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="rank-decline" style="padding:20px;">
  <div style="font-weight:700; font-size:16px; color:#b71c1c; margin-bottom:8px;">
    ⚠️ The False Positive: Ravi Mehta
  </div>
  <div style="font-size:28px; font-weight:800; color:#c62828;">
    ATS: #{ravi_ats} → TalentForge: #{ravi_tf}
  </div>
  <div style="font-size:13px; color:#e53935; margin-top:8px;">
    1-year internship. Zero shipped products. Resume packed with every
    ML buzzword known. ATS ranked him top-5. TalentForge wasn't fooled.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>")

    # JD info
    with st.expander("📄 Job Description Used"):
        st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)

    # Render full results tabs
    render_results_tabs(
        results=results,
        baseline_results=baseline_results,
        explanation_cards=explanation_cards,
        recruiter_memo=recruiter_memo,
        evaluation=evaluation,
        candidates_meta=candidates_meta,
        jd_schema=jd_schema,
    )

    # Download
    st.markdown("---")
    _render_download_csv(results)


# ──────────────────────────────────────────────────────────────────────────────
# Main router
# ──────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    page = st.session_state.page

    if page == "Home":
        render_home()
    elif page == "Run Analysis":
        render_run_analysis()
    elif page == "Demo Mode":
        render_demo_mode()


if __name__ == "__main__":
    main()
