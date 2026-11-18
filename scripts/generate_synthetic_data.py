"""
scripts/generate_synthetic_data.py — Generate 60 diverse synthetic candidate profiles.

Creates realistic candidate JSON files for TalentForge demo and evaluation.
The dataset is carefully designed to highlight TalentForge's advantages over
keyword-only ATS systems, especially for hidden gems like Arjun Sharma.

Run: python scripts/generate_synthetic_data.py
"""

import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "synthetic_candidates")

# ──────────────────────────────────────────────────────────────────────────────
# Candidate definitions (60 total)
# ──────────────────────────────────────────────────────────────────────────────

CANDIDATES = []

# ============================================================================
# GROUP 1: Strong Senior ML Engineers (10) — obvious top candidates
# These should rank top-10 on BOTH ATS and TalentForge
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_001",
        "name": "Priya Krishnamurthy",
        "ground_truth_quality": "high",
        "ground_truth_notes": "8 years ML experience, led recommendation system at Swiggy serving 20M users, strong PyTorch and MLOps skills, multiple publications.",
        "candidate_type": "strong_senior",
        "resume_text": """PRIYA KRISHNAMURTHY
Senior Machine Learning Engineer
priya.k@email.com | LinkedIn: linkedin.com/in/priyak | Bangalore, India

PROFESSIONAL SUMMARY
8 years of experience designing and deploying large-scale ML systems in production.
Specializes in recommendation systems, ranking models, and ML infrastructure.
Led teams of 4-6 engineers across Swiggy and Flipkart.

EXPERIENCE

Senior ML Engineer — Swiggy (Bangalore)
January 2021 – Present (3.5 years)
- Architected and deployed a real-time restaurant recommendation system serving 20M+ daily active users using a two-stage retrieve-and-rank pipeline
- Reduced recommendation latency from 180ms to 42ms through Triton Inference Server optimization and model distillation
- Built an automated model retraining pipeline with MLflow + Airflow that cut model refresh cycles from 2 weeks to 6 hours
- Mentored 4 junior engineers; led weekly ML platform sync attended by 20+ stakeholders
- Tech stack: PyTorch, Feast feature store, AWS SageMaker, MLflow, Airflow, Docker, Kubernetes, Redis

ML Engineer — Flipkart (Bangalore)
June 2018 – December 2020 (2.5 years)
- Built a real-time bidding ML model for Flipkart Ads, improving click-through rate by 23% over the previous rule-based system
- Designed A/B testing framework for 15+ ML experiments simultaneously; wrote internal documentation adopted company-wide
- Implemented feature engineering pipelines using Apache Spark processing 2TB daily
- Tech stack: TensorFlow, scikit-learn, Spark, SQL, Hive, GCP BigQuery

Data Scientist — Mu Sigma (Bangalore)
July 2016 – May 2018 (2 years)
- Delivered predictive churn model for a US telecom client achieving 87% precision at 20% recall threshold
- Built customer segmentation pipeline used by the client's marketing team across 8M customer records

EDUCATION
B.Tech in Computer Science — IIT Madras, 2016 (CGPA: 9.1/10)

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, MLflow, Airflow, Docker, Kubernetes,
AWS SageMaker, GCP Vertex AI, Apache Spark, Feast, Triton Inference Server,
SQL, Redis, Kafka, FAISS, recommendation systems, ranking models, A/B testing

PROJECTS
- RecSys@Scale: Open-source two-tower recommendation architecture benchmarked on MovieLens-20M
- MLOps Template: GitHub repository with 500+ stars for production ML pipeline templates

PUBLICATIONS
- "Efficient Real-Time Recommendation at Scale" — RecSys 2022 (co-author)
""",
    },
    {
        "id": "candidate_002",
        "name": "Rahul Anand",
        "ground_truth_quality": "high",
        "ground_truth_notes": "7 years ML, strong NLP focus, BERT fine-tuning expertise, led team at Zoho, AWS certified.",
        "candidate_type": "strong_senior",
        "resume_text": """RAHUL ANAND
Senior ML Engineer | NLP Specialist
rahul.anand@email.com | Chennai, India

SUMMARY
7 years building NLP systems at scale. Expert in transformer fine-tuning, information extraction,
and conversational AI. Currently leading NLP infrastructure at Zoho for 50M+ user products.

EXPERIENCE

Senior ML Engineer — Zoho Corporation (Chennai)
March 2020 – Present (4 years)
- Built multilingual intent classification system for Zoho Desk supporting 12 languages,
  serving 50M+ queries/day with 94.3% accuracy
- Fine-tuned BERT-based models for contract NLP: achieved 91% F1 on named entity recognition,
  replacing a rule-based system that had been in production for 6 years
- Designed ML serving layer using FastAPI + Docker + Kubernetes; cut inference costs by 40%
  through dynamic batching and model quantization (INT8)
- Led a 3-person NLP team; established MLflow-based experiment tracking across the org

ML Engineer — Freshworks (Chennai)
August 2017 – February 2020 (2.5 years)
- Built sentiment analysis pipeline processing 500K support tickets/day using DistilBERT
- Implemented active learning loop reducing labeling cost by 60% on a customer intent dataset
- Created Python SDK for ML model serving adopted by 5 internal product teams

Junior Data Scientist — Cognizant (Chennai)
July 2015 – July 2017 (2 years)
- Developed credit scoring models for a UK banking client using XGBoost (AUC: 0.89)
- Built automated data validation pipeline reducing data quality issues by 70%

EDUCATION
M.Tech in Data Science — IIT Madras, 2015
B.E. in Electronics — Anna University, 2013

SKILLS
Python, PyTorch, HuggingFace Transformers, BERT, DistilBERT, FastAPI,
Docker, Kubernetes, MLflow, scikit-learn, SQL, AWS (SageMaker, Lambda, S3),
Kafka, Redis, NLP, information extraction, conversational AI, model quantization

CERTIFICATIONS
AWS Certified Machine Learning – Specialty (2022)
""",
    },
    {
        "id": "candidate_003",
        "name": "Anika Mehrotra",
        "ground_truth_quality": "high",
        "ground_truth_notes": "9 years, built ML platform at Amazon India, strong MLOps, led team of 8.",
        "candidate_type": "strong_senior",
        "resume_text": """ANIKA MEHROTRA
Staff ML Engineer | ML Platform Specialist
anika.m@email.com | Delhi, India

SUMMARY
9 years in ML engineering with focus on ML platform infrastructure. Built
Amazon India's ML feature platform used by 200+ data scientists. Expert in
scalable MLOps, feature stores, and model serving at internet scale.

EXPERIENCE

Staff ML Engineer — Amazon India (Delhi)
January 2019 – Present (5 years)
- Designed and built Amazon India's internal ML feature platform (similar to Uber Michelangelo),
  supporting 200+ data scientists and processing 50TB of feature data daily
- Led migration of 80+ legacy models from on-premise servers to AWS SageMaker,
  cutting operational overhead by 60% and model deployment time from 3 days to 2 hours
- Built automated model quality gates: statistical tests, bias checks, and performance
  regression detection that prevented 12 incidents from reaching production
- Managed team of 8 engineers; conducted quarterly platform architecture reviews
- Tech stack: AWS SageMaker, Feast, MLflow, Airflow, Spark, Docker, Kubernetes, Python, Java

Senior ML Engineer — Paytm (Noida)
June 2016 – December 2018 (2.5 years)
- Built fraud detection system processing 10M+ daily transactions with 99.2% uptime SLA
- Implemented online learning pipeline updating fraud model weights every 15 minutes
- Reduced false positive rate by 35% compared to previous threshold-based system

Data Scientist — EXL Analytics (Noida)
July 2015 – May 2016 (1 year)
- Built predictive models for insurance claim propensity for US insurance clients

EDUCATION
B.Tech in Computer Science — IIT Delhi, 2015 (CGPA: 9.3)

SKILLS
Python, AWS SageMaker, Feast, MLflow, Apache Airflow, Apache Spark,
Docker, Kubernetes, PyTorch, TensorFlow, scikit-learn, SQL, Java,
model serving, feature stores, ML platform engineering, fraud detection

PROJECTS
- FeatureHub: Internal open-source feature store (now used by 3 external companies)
""",
    },
    {
        "id": "candidate_004",
        "name": "James Okonkwo",
        "ground_truth_quality": "high",
        "ground_truth_notes": "7 years ML/CV, strong computer vision background, shipped models to 10M users.",
        "candidate_type": "strong_senior",
        "resume_text": """JAMES OKONKWO
Senior Machine Learning Engineer — Computer Vision
james.o@email.com | Pune, India (originally Lagos, Nigeria)

SUMMARY
7 years building production computer vision systems. Led CV engineering at
HealthTech startup that grew from 5 to 200 employees. Deep expertise in
model optimization for edge deployment and medical imaging.

EXPERIENCE

Senior ML Engineer — Niramai Health Analytix (Bangalore)
February 2020 – Present (4 years)
- Built breast cancer screening model using thermal imaging; achieved 89.5% sensitivity
  at 85% specificity — FDA submission package pending
- Optimized model for edge deployment on Raspberry Pi (TensorRT, FP16 quantization),
  enabling use in rural health clinics without reliable internet
- Built complete MLOps pipeline: DICOM data ingestion → augmentation → training → validation
  → DICOM RT output, reducing radiologist review time by 40%
- Collaborated with 3 radiologists to build ground truth annotation pipeline

ML Engineer — Staqu Technologies (Noida)
January 2018 – January 2020 (2 years)
- Built person re-identification system for smart city surveillance (mAP: 0.78)
- Deployed face recognition pipeline processing 500K faces/day across 200+ cameras
- Tech stack: PyTorch, OpenCV, TensorRT, FastAPI, Docker, NVIDIA Jetson

Junior ML Engineer — Persistent Systems (Pune)
July 2017 – December 2017 (0.5 years)
- Built image classification models for defect detection in automotive parts

EDUCATION
M.Tech in Computer Vision — IIT Bombay, 2017
B.Sc in Computer Science — University of Lagos, 2015

SKILLS
Python, PyTorch, TensorFlow, OpenCV, TensorRT, ONNX, scikit-learn,
Docker, Kubernetes, FastAPI, NVIDIA Jetson, AWS (SageMaker, S3, Lambda),
computer vision, object detection, medical imaging, model quantization, edge deployment

PROJECTS
- MedSegNet: Open-source medical image segmentation toolkit (GitHub: 800+ stars)
""",
    },
    {
        "id": "candidate_005",
        "name": "Sunita Patel",
        "ground_truth_quality": "high",
        "ground_truth_notes": "6 years, recommendation systems at Hotstar, strong distributed systems, excellent leadership.",
        "candidate_type": "strong_senior",
        "resume_text": """SUNITA PATEL
Senior ML Engineer — Recommendations & Personalization
sunita.p@email.com | Mumbai, India

SUMMARY
6 years of ML engineering with focus on content recommendation and personalization
at Hotstar (Disney+ India). Led the recommendation team serving 500M+ users during
IPL seasons. Expert in large-scale collaborative filtering and session-based recommendation.

EXPERIENCE

Senior ML Engineer — Disney+ Hotstar (Mumbai)
April 2020 – Present (4 years)
- Led content recommendation for Disney+ Hotstar (500M+ users during IPL),
  owning the real-time candidate generation → ranking pipeline
- Built session-based transformer model for next-video recommendation;
  increased watch time by 18% over previous collaborative filtering baseline
- Architected model serving infrastructure handling 200K QPS at peak (IPL finals)
  using TorchServe + Redis + Kubernetes HPA
- Built automated A/B experimentation platform for recommendation experiments
  running 30+ simultaneous tests
- Mentored team of 5 engineers across 2 time zones

ML Engineer — Jio Platforms (Mumbai)
June 2018 – March 2020 (1.75 years)
- Built music recommendation engine for JioSaavn using matrix factorization
  (ALS, SVD++) achieving 22% CTR improvement
- Designed real-time feature pipeline using Kafka + Flink for 50M daily active users

Data Scientist — LatentView Analytics (Chennai)
July 2017 – May 2018 (1 year)
- Delivered customer lifetime value model for US retail client (R² = 0.87)

EDUCATION
B.Tech in Computer Science — BITS Pilani, 2017 (CGPA: 9.0)

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, TorchServe, Kafka, Apache Flink,
Redis, Docker, Kubernetes, SQL, AWS, collaborative filtering, session-based recommendation,
transformer models, A/B testing, recommendation systems, content personalization
""",
    },
    {
        "id": "candidate_006",
        "name": "Marcus Chen",
        "ground_truth_quality": "high",
        "ground_truth_notes": "8 years, Google Brain research background, strong theory + practice, NeurIPS papers.",
        "candidate_type": "strong_senior",
        "resume_text": """MARCUS CHEN
Senior Research ML Engineer
marcus.c@email.com | Bangalore, India (originally Beijing, China)

SUMMARY
8 years combining ML research and engineering. Published 3 NeurIPS papers.
Led applied research team at Google Brain India. Expert in deep learning
theory, optimization, and translating research into production systems.

EXPERIENCE

Senior Research Engineer — Google Brain India (Bangalore)
July 2019 – Present (5 years)
- Led applied research on efficient transformers; core contributor to the
  FNet architecture (published NeurIPS 2021)
- Built distributed training system for 10B+ parameter models using JAX + TPUs
- Translated 4 research papers into production ML features shipped to Google Search
  and Google Translate, impacting 1B+ users
- Managed team of 4 research engineers; organized quarterly research reviews

ML Research Engineer — Microsoft Research India (Bangalore)
June 2016 – June 2019 (3 years)
- Built neural machine translation system for Indic languages (Hindi, Tamil, Telugu)
  with BLEU score of 38.2, improving over then-state-of-the-art by 4.1 points
- Implemented custom CUDA kernels for attention mechanism optimization (2x speedup)
- Published 2 papers at ACL 2018 and EMNLP 2019

EDUCATION
M.S. in Computer Science (Machine Learning) — Stanford University, 2016
B.Sc. in Mathematics — Peking University, 2014

SKILLS
Python, JAX, PyTorch, TensorFlow, CUDA, distributed training, TPUs,
transformers, NLP, optimization, scikit-learn, Docker, Kubernetes,
GCP (Vertex AI, TPU pods), SQL, research methodology, A/B testing

PUBLICATIONS
- FNet (NeurIPS 2021), two ACL/EMNLP papers on Indic NMT
""",
    },
    {
        "id": "candidate_007",
        "name": "Lakshmi Venkataraman",
        "ground_truth_quality": "high",
        "ground_truth_notes": "6 years, strong MLOps + platform engineering, built end-to-end ML platform at Ola.",
        "candidate_type": "strong_senior",
        "resume_text": """LAKSHMI VENKATARAMAN
Senior ML Platform Engineer
lakshmi.v@email.com | Bangalore, India

SUMMARY
6 years building ML infrastructure platforms. Architect of Ola's ML Platform
serving 30+ data scientists. Deep expertise in Kubernetes-native ML workflows,
feature engineering at scale, and automated model governance.

EXPERIENCE

Senior ML Platform Engineer — Ola Electric (Bangalore)
March 2021 – Present (3 years)
- Designed and built ML platform for Ola Electric's battery health prediction system
  processing telemetry from 500K+ electric vehicles in real time
- Built automated retraining pipeline triggered by drift detection (KL-divergence monitoring)
  — reduced model degradation incidents by 80%
- Implemented model governance framework: lineage tracking, bias detection, explanation reports
  (SHAP values) for all production models
- Built Kubeflow-based pipeline that reduced model deployment from 1 week to 4 hours

Senior Data Engineer — Ola (Bangalore)
January 2019 – February 2021 (2 years)
- Built real-time ETA prediction data pipeline using Kafka + Spark Streaming
  processing 2M ride events/minute
- Designed Feast feature store serving 50+ ML models with sub-5ms latency

Data Engineer — Mu Sigma (Bangalore)
July 2017 – December 2018 (1.5 years)
- Built ETL pipelines for Fortune 500 retail and telecom clients

EDUCATION
B.Tech in Computer Science — NIT Trichy, 2017 (CGPA: 8.9)

SKILLS
Python, Kubernetes, Kubeflow, Apache Airflow, MLflow, Feast, Apache Kafka,
Apache Spark, Docker, FastAPI, Prometheus, Grafana, Evidently AI, AWS,
SHAP, scikit-learn, PyTorch, SQL, model governance, feature stores, MLOps
""",
    },
    {
        "id": "candidate_008",
        "name": "Rohan Kapoor",
        "ground_truth_quality": "high",
        "ground_truth_notes": "7 years, fintech ML, fraud detection and risk scoring expert, strong Python + Spark.",
        "candidate_type": "strong_senior",
        "resume_text": """ROHAN KAPOOR
Senior Data Scientist — Risk & Fraud ML
rohan.k@email.com | Gurgaon, India

SUMMARY
7 years building ML systems for financial risk at HDFC Bank and Razorpay.
Led fraud detection team that saved ₹420 crore in prevented fraud annually.
Expert in real-time scoring, graph neural networks for fraud rings, and
model monitoring in high-stakes financial environments.

EXPERIENCE

Senior Data Scientist — Razorpay (Bangalore)
September 2020 – Present (3.5 years)
- Led fraud ML team; built real-time transaction fraud detection system
  processing 5M+ transactions/day with sub-20ms latency
- Designed graph neural network model detecting fraud rings by analyzing
  merchant-customer-device relationship graphs (reduced chargebacks by 34%)
- Built automated model retraining pipeline with custom drift detection for
  concept drift in fraud patterns — reduced manual monitoring effort by 70%
- Collaborated with Risk, Legal, and Engineering to build model explanation
  system meeting RBI regulatory requirements (SHAP-based)

ML Engineer — HDFC Bank (Mumbai)
July 2017 – August 2020 (3 years)
- Built credit underwriting ML model for HDFC personal loans
  (Gini coefficient: 0.72, improved from 0.61 on previous scorecard)
- Designed ETL pipeline for credit bureau data integration (CIBIL, Experian)
- Built customer propensity models for cross-sell (14% response rate uplift)

EDUCATION
MBA (Finance + Analytics) — IIM Calcutta, 2017
B.Tech in Electrical Engineering — IIT Kanpur, 2015

SKILLS
Python, PyTorch, XGBoost, LightGBM, scikit-learn, SHAP, SQL, Apache Spark,
Kafka, AWS, Docker, MLflow, graph neural networks, fraud detection,
credit scoring, risk modeling, A/B testing, model monitoring

CERTIFICATIONS
FRM (Financial Risk Manager), GARP — 2019
""",
    },
    {
        "id": "candidate_009",
        "name": "Sarah Oduya",
        "ground_truth_quality": "high",
        "ground_truth_notes": "6 years, NLP expert, built search ranking at IndiaMart, strong semantic search.",
        "candidate_type": "strong_senior",
        "resume_text": """SARAH ODUYA
Senior ML Engineer — Search & NLP
sarah.o@email.com | Noida, India (originally Nairobi, Kenya)

SUMMARY
6 years in NLP and information retrieval. Built semantic search and
product ranking systems at IndiaMart serving 100M+ buyers/month.
Expert in dense retrieval, reranking, and multilingual NLP.

EXPERIENCE

Senior ML Engineer — IndiaMart (Noida)
June 2020 – Present (4 years)
- Rebuilt product search from BM25 to a dense retrieval + cross-encoder reranking pipeline;
  improved search relevance (NDCG@10: 0.54 → 0.71) and increased query-to-contact rate by 28%
- Built multilingual query understanding system handling Hindi, Marathi, Gujarati queries
  using mBERT fine-tuned on 2M IndiaMart search logs
- Designed A/B testing infrastructure for search experiments;
  ran 40+ experiments in 2023 alone
- Tech stack: PyTorch, HuggingFace, FAISS, Elasticsearch, FastAPI, Kubernetes, GCP

ML Engineer — MakeMyTrip (Gurgaon)
July 2018 – May 2020 (2 years)
- Built hotel search ranking model combining content similarity, review sentiment,
  and booking probability scores
- Developed review summarization system using extractive NLP for 10M+ hotel reviews

EDUCATION
M.S. in Computer Science — University of Nairobi, 2018
B.Sc. in Mathematics and CS — Kenyatta University, 2016

SKILLS
Python, PyTorch, HuggingFace Transformers, FAISS, Elasticsearch, FastAPI,
Docker, Kubernetes, GCP, BM25, dense retrieval, cross-encoder, BERT, mBERT,
NLP, information retrieval, multilingual models, search ranking, A/B testing
""",
    },
    {
        "id": "candidate_010",
        "name": "Vikram Nair",
        "ground_truth_quality": "high",
        "ground_truth_notes": "9 years, ML generalist with strong leadership, built ML orgs at two startups.",
        "candidate_type": "strong_senior",
        "resume_text": """VIKRAM NAIR
Staff Machine Learning Engineer
vikram.n@email.com | Bangalore, India

SUMMARY
9 years leading ML teams at high-growth startups. Built ML capability from
zero to 15-person teams at two companies. Deep experience across recommendation,
NLP, and MLOps. Strong engineering and product thinking.

EXPERIENCE

Staff ML Engineer — Meesho (Bangalore)
April 2021 – Present (3 years)
- Built Meesho's ML platform serving 50M+ monthly active users across
  product recommendation, pricing, and logistics
- Led hiring and technical growth of ML team from 5 to 15 engineers
- Architected the real-time recommendation system combining collaborative filtering
  (two-tower model) with session context — drove 31% increase in order conversion
- Built model governance and explainability layer required for RBI and TRAI compliance

Senior ML Engineer — Urban Company (Gurgaon)
September 2018 – March 2021 (2.5 years)
- Built service professional matching ML system; reduced booking cancellation by 22%
- Designed dynamic pricing ML model for Urban Company services

ML Engineer — Ola (Bangalore)
June 2015 – August 2018 (3.25 years)
- Built ETA prediction system for Ola rides using gradient boosting + contextual features
- Led 3-person ML team responsible for ride matching optimization

EDUCATION
B.Tech in Computer Science — BITS Goa, 2015 (CGPA: 8.8)

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, MLflow, Airflow, Docker,
Kubernetes, AWS, Kafka, Redis, FAISS, SQL, recommendation systems,
dynamic pricing, matching optimization, ML platform engineering, team leadership
""",
    },
]

# ============================================================================
# GROUP 2: Mid-level with strong projects but weak keyword coverage (10)
# The "Hidden Gem" candidates — ranked low by ATS, high by TalentForge
# Arjun Sharma is the demo protagonist (candidate_011)
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_011",
        "name": "Arjun Sharma",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, led deployment of ML system serving 2M users at a 15-person startup. Uses 'ML infrastructure' instead of MLOps, 'model training pipeline' instead of CI/CD. ATS misses this candidate completely. TalentForge correctly ranks ~2nd.",
        "candidate_type": "good_hidden",
        "resume_text": """ARJUN SHARMA
ML Infrastructure Engineer
arjun.sharma@protonmail.com | Jaipur, India

ABOUT ME
I build the plumbing that lets ML teams ship models to production. At Krishi Network,
a 15-person agritech startup, I took our ML capabilities from one-off notebooks to
a fully automated model training pipeline that now serves 2 million farmers daily
for crop disease detection and yield prediction.

WORK HISTORY

ML Infrastructure Engineer — Krishi Network (Jaipur / Remote)
October 2020 – Present (3.5 years)
- Designed and built the entire ML infrastructure for a crop disease detection system
  from ground up: data collection → preprocessing → model training pipeline →
  automated testing → production serving → monitoring dashboard
- System now runs models serving 2M farmers across Rajasthan and Gujarat
  with 99.4% uptime over the last 18 months
- Built automated model validation pipeline: each new model version runs
  statistical equivalence tests against production before deployment —
  no manual intervention needed for routine updates
- Implemented concept drift monitoring that pages on-call team when
  seasonal crop patterns shift — caught 3 model degradations before
  they affected farmers
- Reduced model deployment cycle from manual 2-week process to
  fully automated 4-hour pipeline
- Built Python API layer wrapping our models, integrated by 2 third-party
  NGOs serving additional 500K users
- Tech: Python, FastAPI, PostgreSQL, Celery, Redis, scikit-learn, OpenCV,
  Nginx, Linux servers (self-managed), Grafana, custom Python monitoring

Junior Backend Developer — TechMahindra (Jaipur)
August 2019 – September 2020 (1.2 years)
- Developed REST APIs for enterprise HR management software using Django
- Built data migration scripts for client database upgrades

SIDE PROJECTS
- AgriVision: Mobile app with embedded TensorFlow Lite model for leaf disease
  detection — 50K+ downloads, 4.3 stars on Play Store
- FarmData API: Open REST API aggregating weather, soil, and crop price data
  used by 3 startups (GitHub: 200+ stars)

EDUCATION
B.Tech in Computer Science — Rajasthan Technical University, 2019 (CGPA: 8.4)

SKILLS
Python, FastAPI, scikit-learn, PyTorch (learning), TensorFlow Lite, OpenCV,
PostgreSQL, Redis, Celery, Grafana, Linux, Docker (basic), Nginx,
model deployment, automated testing, monitoring, REST APIs

NOTE: I am self-taught in most of my ML infrastructure skills — I learned by doing.
I am currently upskilling in Kubernetes and Airflow.
""",
    },
    {
        "id": "candidate_012",
        "name": "Meera Joshi",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, built production NLP system at small company, uses non-standard terminology, very strong projects.",
        "candidate_type": "good_hidden",
        "resume_text": """MEERA JOSHI
Python Developer specializing in language AI
meera.j89@gmail.com | Pune, India

EXPERIENCE

Python Developer — DocuSmart Technologies (Pune)
January 2021 – Present (3.5 years)
- Built text extraction and classification engine for legal documents
  that reduced manual document review time by 65% for our 40 law firm clients
- Trained custom BERT-based model to identify 22 categories of legal clauses
  (F1: 0.88) using labeled data from 5,000 contracts
- Designed the entire model development lifecycle: raw PDF ingestion,
  annotation pipeline, training scripts, model evaluation, API deployment
- System processes 10,000+ legal documents per month with 4 clients now
  using it as their primary workflow tool
- Built custom named entity recognition model for identifying parties,
  dates, obligations in contracts
- Tech: Python, PyTorch, HuggingFace, FastAPI, PostgreSQL, Celery, Docker

Junior Developer — Infosys (Pune)
August 2019 – December 2020 (1.3 years)
- Built internal tools for HR data processing and reporting
- Wrote Python scripts for automating client data transformation tasks

EDUCATION
M.Sc. in Computer Science — Pune University, 2019

SKILLS
Python, PyTorch, HuggingFace Transformers, FastAPI, PostgreSQL, Docker,
NLP, text classification, named entity recognition, BERT, legal tech,
scikit-learn, pandas, matplotlib

PROJECTS
- ContractIQ: Side project — NLP tool for startup contract review,
  used by 200+ founders (Product Hunt: #5 of the day)
""",
    },
    {
        "id": "candidate_013",
        "name": "Rahul Biswas",
        "ground_truth_quality": "high",
        "ground_truth_notes": "5 years, built predictive maintenance system for manufacturing, strong ML engineering but not standard ML buzzwords.",
        "candidate_type": "good_hidden",
        "resume_text": """RAHUL BISWAS
Software Engineer — Predictive Systems
rahul.biswas@outlook.com | Kolkata, India

PROFESSIONAL BACKGROUND
I work on software that predicts equipment failures before they happen.
At Tata Steel, I built a predictive maintenance system that now monitors
800+ industrial machines 24/7 and has prevented 47 unplanned shutdowns
worth approximately ₹180 crore in avoided downtime.

WORK HISTORY

Software Engineer — Tata Steel Digital (Kolkata)
April 2019 – Present (5 years)
- Built machine learning system to predict bearing failures in steel mill
  machinery using vibration sensor data (time-series LSTM model)
- Model detects failures 72 hours in advance with 83% precision, 91% recall
- Engineered features from raw sensor streams: FFT features, rolling statistics,
  domain-specific vibration signatures recommended by mechanical engineers
- Automated the full retraining cycle: new sensor data → preprocessing →
  model training → automated validation against holdout set → deployment
  to production monitoring dashboard
- Built alert system integrated with factory floor PLC systems
- Reduced false alarm rate from 31% (previous rule-based system) to 9%
- Tech: Python, scikit-learn, Keras, pandas, numpy, InfluxDB, Grafana,
  FastAPI, Linux, PostgreSQL

Junior Software Engineer — Wipro (Kolkata)
July 2018 – March 2019 (0.75 years)
- Developed Java microservices for enterprise logistics tracking software

EDUCATION
B.Tech in Electronics and Communication — Jadavpur University, 2018 (CGPA: 8.7)

SKILLS
Python, scikit-learn, Keras/TensorFlow, pandas, numpy, time-series analysis,
LSTM, feature engineering, InfluxDB, Grafana, FastAPI, PostgreSQL, Java, Linux,
predictive maintenance, anomaly detection, industrial IoT
""",
    },
    {
        "id": "candidate_014",
        "name": "Ananya Singh",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, recommendation and personalization at edtech startup, strong projects, non-standard job titles.",
        "candidate_type": "good_hidden",
        "resume_text": """ANANYA SINGH
Backend Engineer — Learning Personalization
ananya.s@proton.me | Lucknow, India

WORK EXPERIENCE

Backend Engineer — Vedantu (Bangalore / Remote)
January 2021 – Present (3.5 years)
- Built the personalized learning recommendation system for Vedantu's K-12 platform
  serving 500K active students
- System recommends practice problems and video lectures based on student performance
  history using collaborative filtering and knowledge tracing models
- Improved student retention metric by 19% (as measured in A/B test over 3 months)
- Built the A/B experiment framework used by the product team for
  all recommendation experiments
- Implemented real-time student performance tracking pipeline using Kafka + Redis
  that powers the adaptive assessment feature
- Tech: Python, scikit-learn, PyTorch, FastAPI, Kafka, Redis, PostgreSQL, Docker

Software Developer — Byju's (Bangalore)
June 2019 – December 2020 (1.5 years)
- Developed backend APIs for the Byju's mobile learning app
- Wrote content ingestion pipeline for 10,000+ learning modules

EDUCATION
B.Tech in Information Technology — VIT Vellore, 2019 (CGPA: 8.6)

SKILLS
Python, PyTorch, scikit-learn, FastAPI, Kafka, Redis, PostgreSQL,
Docker, SQL, collaborative filtering, knowledge tracing, A/B testing,
recommendation systems, edtech, personalization

HOBBY PROJECTS
- StudyPal: WhatsApp bot that recommends daily revision using spaced repetition algorithm
  — 8,000 active users across 3 Telegram groups
""",
    },
    {
        "id": "candidate_015",
        "name": "Emmanuel Adebayo",
        "ground_truth_quality": "high",
        "ground_truth_notes": "5 years, strong healthcare ML, different terminology than standard ML role.",
        "candidate_type": "good_hidden",
        "resume_text": """EMMANUEL ADEBAYO
Data Engineer and ML Developer
e.adebayo@email.com | Hyderabad, India (originally Lagos, Nigeria)

I build machine learning solutions for healthcare diagnostics. My models are used
by doctors in 12 hospitals across Telangana for early disease screening.

EMPLOYMENT

ML and Data Engineer — HealthFirst Diagnostics (Hyderabad)
March 2020 – Present (4 years)
- Built diabetic retinopathy screening model using fundus photography
  (sensitivity: 91%, specificity: 88%) — validated against ophthalmologist labels
- System deployed in 12 rural hospitals, screening 3,000+ patients/month
  in areas with no specialist access
- Built the complete data pipeline: image collection → quality filtering →
  preprocessing → model training → evaluation → deployment to clinic tablets
- Designed annotation workflow with 3 consultant ophthalmologists;
  inter-annotator agreement of 0.89 Cohen's kappa
- Reduced average screening cost from ₹800 to ₹45 per patient by replacing
  manual specialist review with AI-assisted triage

Data Analyst — Deloitte India (Hyderabad)
January 2019 – February 2020 (1.1 years)
- Built financial dashboards and automated reports for healthcare client accounts
- Wrote SQL pipelines for data extraction and transformation

EDUCATION
M.Tech in Biomedical Engineering — IIT Hyderabad, 2019
B.Sc. in Computer Science — University of Ibadan, 2016

SKILLS
Python, PyTorch, OpenCV, scikit-learn, FastAPI, PostgreSQL, Pandas,
NumPy, medical imaging, image classification, annotation management,
Linux, Docker, Streamlit (for dashboard), SQL
""",
    },
    {
        "id": "candidate_016",
        "name": "Pooja Tiwari",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, strong time series ML at fintech, non-standard resume format, excellent impact numbers.",
        "candidate_type": "good_hidden",
        "resume_text": """POOJA TIWARI
Quantitative Developer / ML Engineer
pooja.tiwari.quant@gmail.com | Mumbai, India

WHAT I DO
I build machine learning models that predict financial market behavior.
At Edelweiss, my models run live in production for algorithmic trading
strategies managing ₹200 crore in assets.

WHERE I'VE WORKED

Quantitative Developer — Edelweiss Securities (Mumbai)
April 2020 – Present (4 years)
- Built LSTM-based stock return prediction models for Indian equity markets
  (used in intraday momentum strategies, Sharpe ratio: 1.42 out of sample)
- Developed options pricing model incorporating ML-predicted volatility surface;
  reduced hedging errors by 28% compared to Black-Scholes baseline
- Built real-time feature engineering pipeline consuming NSE tick data
  (100K+ ticks/second) using Kafka + Python consumer workers
- Created model performance monitoring dashboard tracking P&L attribution
  by model component using Grafana

Analyst — ICICI Prudential AMC (Mumbai)
July 2019 – March 2020 (0.75 years)
- Built mutual fund performance attribution analytics using Python and Excel

EDUCATION
B.Sc. in Mathematics and Statistics — Mumbai University, 2019 (First Class, 83%)

SKILLS
Python, pandas, numpy, scikit-learn, Keras/TensorFlow, Kafka, Grafana,
PostgreSQL, SQL, LSTM, time series forecasting, feature engineering,
quantitative finance, options pricing, momentum strategies
""",
    },
    {
        "id": "candidate_017",
        "name": "Nadia Kamau",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, strong NLP hidden gem, built text analytics at a non-tech company.",
        "candidate_type": "good_hidden",
        "resume_text": """NADIA KAMAU
Python & Machine Learning Developer
nadia.k@email.com | Bangalore, India (originally Nairobi, Kenya)

WORK EXPERIENCE

Python Developer — Times of India Digital (Bangalore)
February 2021 – Present (3.5 years)
- Built automatic article categorization and tagging system for Times of India
  digital content team; handles 1,500+ articles/day with 94% accuracy
- Designed recommendation engine for related articles using TF-IDF + BERT
  embeddings; increased page views per session by 23%
- Built news summarization tool using extractive NLP that generates
  2-sentence summaries — used by editorial team for breaking news
- Created content duplication detection system preventing republication
  of similar articles (cosine similarity threshold on BERT embeddings)
- Tech: Python, HuggingFace, scikit-learn, FastAPI, MongoDB, Redis, Docker

Data Analyst — Accenture India (Bangalore)
June 2019 – January 2021 (1.5 years)
- Built BI dashboards for telecom client using SQL and Tableau
- Automated monthly reporting using Python scripts saving 8 hours/week

EDUCATION
B.E. in Computer Science — RV College of Engineering, Bangalore, 2019

SKILLS
Python, HuggingFace Transformers, BERT, scikit-learn, FastAPI,
MongoDB, Redis, Docker, NLP, text classification, summarization,
recommendation systems, content analytics, SQL, Tableau
""",
    },
    {
        "id": "candidate_018",
        "name": "Siddharth Kulkarni",
        "ground_truth_quality": "high",
        "ground_truth_notes": "5 years, logistics ML expert, built route optimization and demand forecasting, non-tech-company background.",
        "candidate_type": "good_hidden",
        "resume_text": """SIDDHARTH KULKARNI
Operations Research / ML Engineer
sid.kulkarni@email.com | Pune, India

PROFESSIONAL SUMMARY
I solve logistics problems with machine learning. At BlueDart, I built demand
forecasting and route optimization systems that handle 200,000 parcels/day
across Maharashtra.

EXPERIENCE

Data Science Lead — BlueDart (Pune)
September 2019 – Present (4.5 years)
- Built parcel demand forecasting model (SARIMA + XGBoost ensemble) for
  200+ delivery zones; reduced hub overstaffing by 18% saving ₹1.2 crore/year
- Implemented dynamic route optimization using ML-predicted traffic conditions;
  reduced average delivery time by 22 minutes per route
- Built real-time exception prediction system for high-risk shipments
  with 82% precision (flagged to ops team for proactive intervention)
- Designed ETL pipeline ingesting GPS, weather, and operational data
  processing 5M events/day
- Tech: Python, scikit-learn, XGBoost, LightGBM, pandas, PostgreSQL,
  Airflow, FastAPI, Tableau

Operations Analyst — Mahindra Logistics (Mumbai)
July 2018 – August 2019 (1 year)
- Analyzed delivery performance data and built KPI dashboards
- Identified route inefficiencies saving ₹40 lakhs in fuel costs

EDUCATION
M.B.A. — SPJIMR Mumbai, 2018
B.E. in Industrial Engineering — COEP Pune, 2016

SKILLS
Python, scikit-learn, XGBoost, LightGBM, pandas, numpy, PostgreSQL,
Apache Airflow, FastAPI, Tableau, SQL, time series forecasting,
optimization, operations research, logistics analytics
""",
    },
    {
        "id": "candidate_019",
        "name": "Aditya Roy",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, strong ML for IoT/edge, semiconductor background, non-standard domain.",
        "candidate_type": "good_hidden",
        "resume_text": """ADITYA ROY
Embedded ML Engineer
aditya.roy.ml@gmail.com | Kolkata, India

I deploy machine learning at the edge — on microcontrollers, FPGAs, and
industrial sensors. At Bosch India, I built the on-device anomaly detection
system used in 50,000 automotive parts manufactured each month.

EXPERIENCE

Embedded ML Engineer — Bosch India (Bangalore)
June 2020 – Present (4 years)
- Built and optimized neural network for anomaly detection in automotive
  brake pad manufacturing; deployed on ARM Cortex-M7 microcontroller
- Achieved 97.3% defect detection rate with <2% false positive rate
- Reduced model size from 4MB to 180KB through quantization (INT8) and
  pruning while maintaining accuracy within 0.5% of floating-point baseline
- Built complete edge ML pipeline: sensor data collection → feature extraction →
  model training (PyTorch) → TFLite conversion → embedded deployment
- Deployed across 8 production lines; system has prevented 340 defective units
  from shipping since deployment

Junior Embedded Engineer — Texas Instruments India (Bangalore)
July 2018 – May 2020 (2 years)
- Developed firmware for DSP signal processing applications
- Wrote C++ libraries for sensor data acquisition on ARM processors

EDUCATION
B.Tech in Electronics Engineering — IIT Kharagpur, 2018 (CGPA: 9.1)

SKILLS
Python, PyTorch, TensorFlow Lite, scikit-learn, C++, ARM Cortex,
quantization, model compression, pruning, edge deployment, ONNX,
OpenCV, NumPy, embedded systems, signal processing, industrial IoT
""",
    },
    {
        "id": "candidate_020",
        "name": "Kavya Reddy",
        "ground_truth_quality": "high",
        "ground_truth_notes": "4 years, social media ML, built content moderation at a startup, impactful but non-enterprise background.",
        "candidate_type": "good_hidden",
        "resume_text": """KAVYA REDDY
ML Engineer — Trust & Safety
kavya.reddy@email.com | Hyderabad, India

EXPERIENCE

ML Engineer — ShareChat (Bangalore / Remote)
May 2020 – Present (4 years)
- Built content moderation ML system for ShareChat's 180M+ user platform
  supporting 15 Indian languages
- Multilingual hate speech detection model (mBERT + language adapters):
  F1 of 0.87 across 8 languages, deployed in production moderating
  500K+ posts/day in real time
- Built NSFW image detection pipeline using EfficientNet; processing
  2M+ images/day with 0.3% false positive rate
- Designed active learning loop that reduced annotation cost by 55%
  while maintaining model quality through smart sample selection
- System prevented estimated 2M+ policy-violating posts from going live in 2023
- Tech: Python, PyTorch, HuggingFace, FastAPI, Kafka, Redis, Docker, Kubernetes

Junior Software Engineer — Mindtree (Hyderabad)
August 2019 – April 2020 (0.75 years)
- Built Java REST APIs for enterprise logistics application

EDUCATION
B.Tech in Computer Science — JNTU Hyderabad, 2019 (CGPA: 8.5)

SKILLS
Python, PyTorch, HuggingFace Transformers, mBERT, FastAPI, Kafka,
Redis, Docker, Kubernetes, NLP, computer vision, EfficientNet,
content moderation, multilingual NLP, active learning, trust and safety
""",
    },
]

# ============================================================================
# GROUP 3: Junior candidates with high growth velocity (10)
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_021",
        "name": "Tanvi Shah",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years experience, excellent academic background IIT, strong project portfolio, high growth velocity.",
        "candidate_type": "high_growth",
        "resume_text": """TANVI SHAH
Junior ML Engineer
tanvi.shah.ml@gmail.com | Ahmedabad, India

EDUCATION
M.Tech in AI — IIT Bombay, 2022 (CGPA: 9.4/10)
Thesis: "Efficient Few-Shot Learning for Low-Resource Languages"
B.Tech in Computer Science — DAIICT, 2020 (CGPA: 9.1/10)

EXPERIENCE

Junior ML Engineer — Groww (Bangalore)
July 2022 – Present (2 years)
- Built mutual fund recommendation model based on user risk profile and
  investment history; A/B tested improvement in investment completion rate (+12%)
- Implemented automated data quality checks catching 95% of upstream data issues
  before they affect model predictions
- Contributed to the ML platform team: wrote internal tooling for model evaluation
  adopted by 6 other data scientists
- Tech: Python, scikit-learn, PyTorch, MLflow, SQL, Docker

ACADEMIC PROJECTS
- Few-shot NLP: Fine-tuned XLM-R for 10 low-resource Indian languages
  with 50 examples each (published at student workshop @ EMNLP 2022)
- RecSys Kaggle: Top 7% finish in Kaggle H&M Recommendation Challenge
- AutoML: Built neural architecture search tool as B.Tech final project

SKILLS
Python, PyTorch, scikit-learn, HuggingFace, MLflow, Docker, SQL,
NumPy, pandas, recommendation systems, NLP, few-shot learning

ACHIEVEMENTS
- IIT Bombay Institute Silver Medal (rank 2 in department)
- Kaggle Expert (6 competitions, 2 silver medals)
""",
    },
    {
        "id": "candidate_022",
        "name": "Aarav Gupta",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2.5 years, fast progression, led a small project team already, strong NLP skills.",
        "candidate_type": "high_growth",
        "resume_text": """AARAV GUPTA
ML Engineer
aarav.gupta@email.com | Delhi, India

EXPERIENCE

ML Engineer — PhonePe (Bangalore)
January 2022 – Present (2.5 years)
- Promoted from Junior to ML Engineer in 14 months (fastest in team history)
- Built spam detection model for PhonePe UPI transactions reducing spam reports by 41%
- Led 2-person sub-team to build merchant category classification (94% accuracy,
  5M+ merchants classified)
- Mentored 1 intern who received full-time return offer

Junior ML Engineer — PhonePe (Bangalore)
July 2021 – December 2021 (0.5 years)
- Joined as campus hire; built data pipelines and model evaluation tools

EDUCATION
B.Tech in Computer Science — Delhi Technological University, 2021 (CGPA: 8.8)

SKILLS
Python, scikit-learn, PyTorch, SQL, Spark (beginner), Docker, MLflow,
NLP, text classification, feature engineering, pandas, numpy

PROJECTS
- PayFraud: Personal project — real-time fraud detection demo using streaming data
- NLP-Hindi: Fine-tuned IndicBERT on Hindi social media data (open sourced)
""",
    },
    {
        "id": "candidate_023",
        "name": "Leila Hosseini",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years, exceptional research background, strong computer vision.",
        "candidate_type": "high_growth",
        "resume_text": """LEILA HOSSEINI
Junior Computer Vision Engineer
leila.h@email.com | Bangalore, India (originally Tehran, Iran)

EDUCATION
M.Sc. Computer Science (Computer Vision) — University of Tehran, 2022 (GPA: 18.9/20)
Thesis: Object detection in low-light conditions using domain adaptation

EXPERIENCE

Computer Vision Engineer — Ather Energy (Bangalore)
October 2022 – Present (2 years)
- Building perception models for Ather electric scooters' advanced safety features
- Implemented lane detection model with 96.2% accuracy for urban Indian roads
- Optimized YOLO-based object detection for onboard GPU (50% latency reduction)
- Co-authored internal technical report on dataset collection methodology

SKILLS
Python, PyTorch, OpenCV, TensorRT, scikit-learn, ONNX,
Docker, CUDA, NumPy, pandas, object detection, semantic segmentation,
domain adaptation, model optimization

PUBLICATIONS
One workshop paper at CVPR 2023 (first author) on nighttime object detection
""",
    },
    {
        "id": "candidate_024",
        "name": "Karan Malhotra",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years, very fast ramp, built a product used by 100K users.",
        "candidate_type": "high_growth",
        "resume_text": """KARAN MALHOTRA
Junior ML Engineer
karan.m.dev@gmail.com | Gurgaon, India

EXPERIENCE

Junior ML Engineer — Razorpay (Bangalore)
August 2022 – Present (2 years)
- Built invoice OCR + information extraction pipeline used by 100K+ merchants
- Improved invoice categorization accuracy from 78% to 91% using fine-tuned
  LayoutLM model on labeled Razorpay merchant data
- Reduced inference cost by 60% through model distillation into smaller BERT
- Demonstrated highest individual model impact in team Q4 review

EDUCATION
B.Tech Computer Science — NIT Warangal, 2022 (CGPA: 9.0)

SKILLS
Python, PyTorch, HuggingFace, LayoutLM, BERT, FastAPI, Docker, SQL,
OCR, document understanding, model distillation, scikit-learn

PROJECTS
- TaxBot: WhatsApp chatbot using GPT-3 for GST queries — 5,000 users
- ResumeParser: Open-source resume parsing library (GitHub: 400+ stars)
""",
    },
    {
        "id": "candidate_025",
        "name": "Shreya Nambiar",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2.5 years, strong data engineering + ML combo, high growth in responsibilities.",
        "candidate_type": "high_growth",
        "resume_text": """SHREYA NAMBIAR
Data Engineer / ML Engineer
shreya.n@email.com | Kochi, India

EXPERIENCE

Data Engineer (transitioning to ML) — Infosys BPM (Bangalore)
January 2021 – Present (3.5 years)
- Started as data engineer, transitioned to ML after 1 year on initiative
- Built customer churn prediction model (self-initiated) now used by
  2 Infosys client accounts to reduce churn by 17%
- Designed Spark-based feature engineering pipeline for ML workflows
- Mentored 2 junior data engineers on Python best practices

EDUCATION
B.Tech in Information Technology — NIT Calicut, 2021

SKILLS
Python, pandas, NumPy, scikit-learn, Apache Spark, SQL, PyTorch (learning),
Docker, Airflow, MLflow (beginner), data engineering, ETL, churn prediction
""",
    },
    {
        "id": "candidate_026",
        "name": "Rohit Pillai",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "3 years, research → industry pivot, strong ML fundamentals.",
        "candidate_type": "high_growth",
        "resume_text": """ROHIT PILLAI
ML Engineer (ex-researcher)
rohit.p.ml@gmail.com | Thiruvananthapuram, India

EXPERIENCE

ML Engineer — Cure.ai (Bangalore)
June 2021 – Present (3 years)
- Built chest X-ray pathology detection model (AUC: 0.91 on NIH ChestX-ray14)
- Improved model robustness through test-time augmentation and ensemble methods
- Built model explanation dashboard using Grad-CAM for radiologists

Research Intern — IISER Pune
May 2020 – May 2021 (1 year)
- Worked on representation learning for biological sequences

EDUCATION
M.Sc. in Statistics — ISI Kolkata, 2020

SKILLS
Python, PyTorch, scikit-learn, OpenCV, medical imaging, Grad-CAM,
NumPy, pandas, statistics, research methodology, Docker
""",
    },
    {
        "id": "candidate_027",
        "name": "Preethi Devi",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years, built ML feature from scratch at startup, high velocity.",
        "candidate_type": "high_growth",
        "resume_text": """PREETHI DEVI
Junior ML Engineer
preethi.d@email.com | Chennai, India

EXPERIENCE

Junior ML Engineer — Licious (Bangalore)
September 2022 – Present (2 years)
- Built fresh produce quality prediction model using image classification
  (ResNet-50, 93% accuracy) reducing customer complaints by 28%
- Deployed model on Android app using TFLite (offline inference)
- Designed data flywheel: new customer feedback automatically feeds
  retraining pipeline

EDUCATION
B.E. Computer Science — CEG Anna University, 2022 (CGPA: 8.9)

SKILLS
Python, PyTorch, TensorFlow Lite, scikit-learn, OpenCV, FastAPI,
Docker, SQL, image classification, mobile ML deployment, pandas

PROJECTS
- FoodSafe: Personal food safety checker app using CV model (Play Store: 2K downloads)
""",
    },
    {
        "id": "candidate_028",
        "name": "Haruto Yamamoto",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years, NLP focus, transferred from Japan, adapting well.",
        "candidate_type": "high_growth",
        "resume_text": """HARUTO YAMAMOTO
Junior ML Engineer
haruto.y@email.com | Bangalore, India (originally Osaka, Japan)

EXPERIENCE

Junior ML Engineer — Turing (Remote / Bangalore)
March 2022 – Present (2.5 years)
- Contributed to LLM fine-tuning pipeline for code generation (Python, Java)
- Built automated evaluation framework for code LLM outputs using unit tests
- Implemented RLHF data collection tool used by 10+ annotators

EDUCATION
M.Sc. in Machine Learning — University of Tokyo, 2022

SKILLS
Python, PyTorch, HuggingFace, LLM fine-tuning, RLHF,
scikit-learn, NumPy, pandas, code generation, evaluation metrics
""",
    },
    {
        "id": "candidate_029",
        "name": "Divya Krishnan",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2 years, strong academic background, built recommendation system at startup.",
        "candidate_type": "high_growth",
        "resume_text": """DIVYA KRISHNAN
Junior ML Engineer
divya.k@email.com | Bangalore, India

EXPERIENCE

ML Engineer — Cred (Bangalore)
August 2022 – Present (2 years)
- Built credit card offer recommendation engine (CTR improvement: +15%)
- Designed offline evaluation harness for recommendation experiments
- Implemented data versioning workflow adopted by entire DS team

EDUCATION
B.Tech in Computer Science — IIT Madras, 2022 (CGPA: 9.2)

SKILLS
Python, PyTorch, scikit-learn, SQL, Docker, MLflow,
recommendation systems, A/B testing, pandas, numpy

PROJECTS
- Kaggle: Top 5% in Recsys Challenge 2022
- IITM RL Club: Built DQN agent for OpenAI CartPole
""",
    },
    {
        "id": "candidate_030",
        "name": "Femi Okonjo",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "2.5 years, strong CV skills, worked on satellite imagery ML.",
        "candidate_type": "high_growth",
        "resume_text": """FEMI OKONJO
Junior Computer Vision Engineer
femi.o@email.com | Bangalore, India (originally Abuja, Nigeria)

EXPERIENCE

Junior ML Engineer — Pixxel Space (Bangalore)
October 2021 – Present (2.5 years)
- Built crop type classification model on multispectral satellite imagery
  (F1: 0.84 across 12 crop types)
- Implemented change detection pipeline for urban expansion monitoring
- Reduced cloud masking false positive rate by 40% using custom model

EDUCATION
M.Sc. in Remote Sensing — IIT Bombay, 2021

SKILLS
Python, PyTorch, scikit-learn, rasterio, GDAL, OpenCV,
remote sensing, satellite imagery, change detection, numpy, pandas

PROJECTS
- Deforestation Alert: Real-time deforestation detection using Sentinel-2 (open source)
""",
    },
]

# ============================================================================
# GROUP 4: Keyword-stuffed candidates (10) — ATS bait, TalentForge detects shallow
# Ravi Mehta is the demo antagonist (candidate_031)
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_031",
        "name": "Ravi Mehta",
        "ground_truth_quality": "low",
        "ground_truth_notes": "1 year internship only, no meaningful projects, resume is dense with buzzwords but zero real impact. Ranks ~2nd on ATS, ~45th on TalentForge. This is the baseline failure example.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """RAVI MEHTA
Machine Learning Engineer | AI/ML Expert | MLOps | LLM | RAG | GenAI
ravi.mehta.ml@gmail.com | Mumbai, India

PROFESSIONAL SUMMARY
Passionate AI/ML professional with expertise in MLOps, LLM fine-tuning, RAG systems,
vector databases, transformer architectures, PyTorch, TensorFlow, Kubernetes, Docker,
CI/CD for ML, model deployment, monitoring, feature stores, and generative AI.
Experienced with LangChain, LlamaIndex, OpenAI GPT-4, Claude, Gemini, Stable Diffusion,
DALL-E, Midjourney API integration, LoRA fine-tuning, QLoRA, PEFT, DPO, RLHF.
Strong background in computer vision, NLP, recommendation systems, fraud detection,
and time series forecasting using state-of-the-art deep learning techniques.

TECHNICAL SKILLS
Languages: Python, R, Scala, Julia, Go (beginner)
ML Frameworks: PyTorch, TensorFlow, JAX, Keras, scikit-learn, XGBoost, LightGBM, CatBoost
LLMs: GPT-4, GPT-4o, Claude 3, Gemini, Mistral, Llama 3, Falcon, Phi-3
MLOps: MLflow, Weights & Biases, Kubeflow, Airflow, DVC, ClearML, Neptune
Vector DBs: FAISS, Pinecone, Chroma, Weaviate, Qdrant, Milvus
Deployment: Docker, Kubernetes, Triton Inference Server, TorchServe, BentoML, FastAPI
Cloud: AWS SageMaker, GCP Vertex AI, Azure ML, AWS Lambda, EC2, S3
Monitoring: Evidently AI, Prometheus, Grafana, Arize AI, WhyLabs
Feature Stores: Feast, Tecton, Hopsworks
Data: Spark, Kafka, Flink, dbt, Airflow, Prefect, Snowflake, BigQuery

EXPERIENCE

AI/ML Intern — StartupXYZ (Mumbai)
June 2023 – May 2024 (1 year)
- Worked on various ML projects involving Python and machine learning
- Used PyTorch and TensorFlow for deep learning experiments
- Explored LLM integration using LangChain and OpenAI API
- Attended company meetings and participated in code reviews
- Helped team with data preprocessing tasks

EDUCATION
B.Tech in Computer Science — Mumbai University, 2023 (CGPA: 6.8)

CERTIFICATIONS
- Machine Learning Specialization — Coursera (Andrew Ng) (2022)
- Deep Learning Specialization — Coursera (2022)
- AWS Cloud Practitioner — 2023
- Google Professional ML Engineer — 2023 (in progress)
- 15+ Udemy certificates in Python, Docker, Kubernetes, MLOps

PROJECTS (from courses and tutorials)
- GPT chatbot using LangChain and OpenAI API (tutorial project)
- MNIST digit classification (standard CNN)
- House price prediction (regression)
- Movie recommendation system (coursework)
- Sentiment analysis on IMDB dataset (NLP tutorial)

ACHIEVEMENTS
- Completed 15+ online ML courses
- Attended 3 AI conferences as attendee
- Member of 5 ML Discord communities
""",
    },
    {
        "id": "candidate_032",
        "name": "Deepak Sharma",
        "ground_truth_quality": "low",
        "ground_truth_notes": "6 months experience, heavily buzzword-padded, no real impact.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """DEEPAK SHARMA
Senior AI Engineer | GenAI | LLM | MLOps | Full Stack ML
deepak.s.ai@gmail.com | Delhi, India

PROFILE
Experienced AI Engineer with comprehensive knowledge of MLOps, CI/CD for machine learning,
Kubernetes, Docker, PyTorch, TensorFlow, LLM fine-tuning with LoRA/QLoRA, RAG pipelines,
vector databases (FAISS, Pinecone, Chroma), recommendation systems, NLP, computer vision,
transformers, BERT, GPT, real-time model serving using Triton and TorchServe, feature
engineering, MLflow, Weights & Biases, Apache Airflow, Kubeflow, model monitoring
with Evidently AI and Prometheus/Grafana. Expert in AWS SageMaker and GCP Vertex AI.

TECHNICAL EXPERTISE
[50-line list of technologies and buzzwords spanning ML, DevOps, cloud, and data engineering]
Python | PyTorch | TensorFlow | scikit-learn | HuggingFace | LangChain | LlamaIndex |
FAISS | Pinecone | Chroma | Weaviate | Docker | Kubernetes | Helm | CI/CD | GitHub Actions |
MLflow | W&B | Kubeflow | Airflow | Prefect | AWS | GCP | Azure | SageMaker | Vertex AI |
Spark | Kafka | Redis | PostgreSQL | MongoDB | Triton | TorchServe | FastAPI | Flask

EXPERIENCE

Data Science Trainee — Self-employed / Freelance
December 2023 – Present (6 months)
- Built personal ML projects and explored AI technologies
- Created chatbots using GPT API and LangChain
- Experimented with Stable Diffusion image generation

EDUCATION
B.Sc. Computer Science — Delhi University, 2023 (CGPA: 6.2)

PROJECTS
- RAG chatbot using LangChain, FAISS, and GPT-4 (tutorial-based)
- Object detection with YOLOv8 (followed YouTube tutorial)
- Fine-tuned Llama-2 7B using QLoRA (Google Colab experiment)
- Built Streamlit dashboard for data visualization

CERTIFICATIONS
20+ online certificates from Coursera, Udemy, YouTube
""",
    },
    {
        "id": "candidate_033",
        "name": "Prashant Kumar",
        "ground_truth_quality": "low",
        "ground_truth_notes": "2 years but entirely non-ML support role, pivoting with a keyword-stuffed CV.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """PRASHANT KUMAR
ML Engineer | Data Scientist | AI Enthusiast
prashant.k.ml@outlook.com | Lucknow, India

SUMMARY
Dynamic and results-oriented ML Engineer with expertise across the entire ML lifecycle
including data collection, feature engineering, model training (PyTorch, TensorFlow,
scikit-learn), MLOps (MLflow, Docker, Kubernetes), model deployment (FastAPI, Triton),
monitoring (Prometheus, Grafana), and cloud platforms (AWS SageMaker, GCP Vertex AI).
Passionate about LLMs, RAG, and Generative AI. Strong knowledge of recommendation systems,
NLP, computer vision, and time series forecasting.

EXPERIENCE

IT Support Engineer — HCL Technologies (Noida)
June 2022 – Present (2 years)
- Resolved Level 1 and Level 2 IT support tickets for enterprise clients
- Managed hardware and software installations for 500+ employees
- Participated in company's AI literacy training program

EDUCATION
B.Tech in Information Technology — AKTU Lucknow, 2022 (CGPA: 6.5)

SKILLS (self-assessed)
Python (intermediate), Machine Learning, Deep Learning, PyTorch, TensorFlow,
MLOps, Docker, Kubernetes, AWS, GCP, NLP, LLMs, RAG, GenAI, SQL, Tableau

PROJECTS
- Sentiment analysis notebook on Kaggle (public kernel, 12 upvotes)
- House price regression using scikit-learn (tutorial)
- GPT-2 text generation experiment (Google Colab)

CERTIFICATIONS
- AI for Everyone — Coursera 2023
- Python for Data Science — Coursera 2022
""",
    },
    {
        "id": "candidate_034",
        "name": "Manish Agarwal",
        "ground_truth_quality": "low",
        "ground_truth_notes": "Overstates experience significantly, keyword-heavy, weak substance.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """MANISH AGARWAL
Lead AI/ML Engineer | 5 Years Experience
manish.agarwal.ai@gmail.com | Jaipur, India

SUMMARY
5+ years experience in ML and AI [Note: 3 years were in digital marketing roles]
Expert in PyTorch, TensorFlow, scikit-learn, MLOps, CI/CD, Docker, Kubernetes,
LLM deployment, RAG systems, vector databases, recommendation systems, NLP, CV.

EXPERIENCE

Digital Marketing Analyst — WeddingWire India (Jaipur)
July 2019 – December 2021 (2.5 years)
- Managed Google Ads and Facebook marketing campaigns
- Analyzed campaign performance data using Excel and basic Python scripts
- Used Google Analytics for website traffic analysis

Junior Data Analyst — Infosys BPO (Jaipur)
January 2022 – December 2022 (1 year)
- Ran SQL queries for business reporting
- Built Excel dashboards for management

ML Engineer (self-declared) — Freelance
January 2023 – Present (1.5 years)
- Built chatbots for small businesses using GPT API
- Created prediction models for personal projects

EDUCATION
B.Com — Rajasthan University, 2019

SKILLS
Python, Machine Learning, Deep Learning, PyTorch, TensorFlow, Docker,
Kubernetes, MLflow, AWS, NLP, computer vision, LLM, RAG, SQL
""",
    },
    {
        "id": "candidate_035",
        "name": "Gaurav Singh",
        "ground_truth_quality": "low",
        "ground_truth_notes": "Fresh graduate with all certifications, no real work, keyword bombs.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """GAURAV SINGH
AI Engineer | MLOps | LLM | Computer Vision | NLP | GenAI
gaurav.singh.ml@gmail.com | Chandigarh, India

PROFESSIONAL SUMMARY
Highly motivated AI/ML Engineer with extensive knowledge of Python, PyTorch, TensorFlow,
HuggingFace, LLM fine-tuning, RAG, vector databases, MLOps, Docker, Kubernetes, CI/CD,
AWS SageMaker, GCP, computer vision, object detection, NLP, recommendation systems,
and Generative AI. Passionate about deploying state-of-the-art AI solutions at scale.

EXPERIENCE: None (fresh graduate)

EDUCATION
B.Tech Computer Science — Chitkara University, 2024 (CGPA: 7.1)

SKILLS
Python, NumPy, pandas, Matplotlib, scikit-learn, PyTorch, TensorFlow, Keras,
HuggingFace, BERT, GPT, LangChain, LlamaIndex, FAISS, Pinecone, Chroma,
Docker, Kubernetes, AWS, GCP, MLflow, Airflow, FastAPI, Flask, SQL, Spark

CERTIFICATIONS (21 total)
- Deep Learning Specialization (Coursera)
- Machine Learning Specialization (Coursera)
- MLOps Specialization (Coursera)
- TensorFlow Developer Certificate
- AWS Solutions Architect Associate
- Multiple Udemy certificates

PROJECTS (all tutorial/course-based)
- Image classification MNIST and CIFAR-10
- GPT chatbot with LangChain
- Movie recommendation with collaborative filtering
- Stock price prediction LSTM
- RAG chatbot with FAISS and OpenAI

GITHUB: 15 repositories, all course exercises
""",
    },
    {
        "id": "candidate_036",
        "name": "Varun Tripathi",
        "ground_truth_quality": "low",
        "ground_truth_notes": "3 years in unrelated role, tiny ML exposure, massive buzzword inflation.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """VARUN TRIPATHI
ML Engineer | Data Scientist | AI Specialist
varun.t.ai@gmail.com | Bhopal, India

CORE COMPETENCIES
Machine Learning, Deep Learning, MLOps, LLMOps, RAG, Vector Databases,
PyTorch, TensorFlow, scikit-learn, XGBoost, Transformers, BERT, GPT-4,
Docker, Kubernetes, Airflow, MLflow, AWS SageMaker, GCP Vertex AI,
Recommendation Systems, NLP, Computer Vision, Fraud Detection, A/B Testing

EXPERIENCE

QA Engineer — Tech Mahindra (Bhopal)
June 2021 – Present (3 years)
- Tested enterprise software applications for banking clients
- Wrote automated test scripts using Selenium and Python
- Participated in agile ceremonies and wrote test plans

EDUCATION
B.Tech Electronics — RGPV Bhopal, 2021 (CGPA: 7.0)

SKILLS (claimed without experience)
Python (for testing), Machine Learning (theoretical), PyTorch, TensorFlow,
Docker (basic), AWS (beginner), SQL

PROJECTS
- Diabetes prediction model (tutorial, Google Colab)
- Simple chatbot using rule-based if/else logic
- Excel dashboard for sales data

CERTIFICATIONS
- Python for Testing — Udemy
- Introduction to Machine Learning — Coursera
- AWS Cloud Practitioner — 2023
""",
    },
    {
        "id": "candidate_037",
        "name": "Nitin Bansal",
        "ground_truth_quality": "low",
        "ground_truth_notes": "1 year data entry experience, inflated ML claims.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """NITIN BANSAL
AI/ML Engineer | Python Developer | Data Scientist
nitin.bansal.ai@gmail.com | Agra, India

TECHNICAL PROFILE
Expert-level skills in Python, Machine Learning, Deep Learning, PyTorch, TensorFlow,
scikit-learn, NLP, Computer Vision, MLOps, Docker, Kubernetes, AWS SageMaker,
GCP Vertex AI, LLM fine-tuning, RAG, FAISS, Pinecone, recommendation systems,
time series forecasting, A/B testing, Airflow, MLflow, Weights & Biases.

EXPERIENCE

Data Entry Operator — Reliance Jio (Agra)
April 2023 – Present (1 year)
- Entered customer data into CRM system
- Prepared daily Excel reports

EDUCATION
B.Sc. Mathematics — Agra University, 2023 (CGPA: 6.0)

PROJECTS (personal, undeployed)
- Linear regression on Boston Housing dataset
- Titanic survival prediction
- Image classification tutorial

CERTIFICATIONS
- Machine Learning A-Z — Udemy 2023
- Python Bootcamp — Udemy 2022
""",
    },
    {
        "id": "candidate_038",
        "name": "Sanjay Rao",
        "ground_truth_quality": "low",
        "ground_truth_notes": "Sales background, pivoting with keyword-stuffed resume.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """SANJAY RAO
Machine Learning Engineer | AI Specialist
sanjay.rao.ml@gmail.com | Hyderabad, India

OBJECTIVE
Highly motivated professional seeking ML Engineer role. Expert in PyTorch, TensorFlow,
Python, scikit-learn, LLMs, RAG, MLOps, Docker, Kubernetes, recommendation systems,
deep learning, NLP, and computer vision.

EXPERIENCE

Sales Executive — Bajaj Allianz (Hyderabad)
August 2021 – Present (3 years)
- Sold life insurance products to retail customers
- Met monthly targets of ₹15 lakh premium collection
- Maintained client relationships and upsold products

EDUCATION
MBA Marketing — Osmania University, 2021
B.Com — Osmania University, 2019

SKILLS
Python (beginner, self-learning), Machine Learning (theoretical),
Excel, PowerPoint, CRM software, communication

COURSES
- Data Science Bootcamp — Scaler Academy (currently enrolled)
- Watched 50+ ML YouTube tutorials

NOTE: "Currently transitioning to ML role after 3 years in sales"
""",
    },
    {
        "id": "candidate_039",
        "name": "Abhishek Jain",
        "ground_truth_quality": "low",
        "ground_truth_notes": "1.5 years in unrelated DevOps role, ML skills purely theoretical.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """ABHISHEK JAIN
ML Engineer | DevOps | Python | AI
abhishek.j.ml@gmail.com | Indore, India

EXPERTISE
Python, Machine Learning, MLOps, CI/CD, Docker, Kubernetes, Jenkins,
PyTorch, TensorFlow, scikit-learn, LLM, RAG, AWS, GCP, Airflow,
MLflow, FastAPI, FAISS, NLP, Computer Vision, Recommendation Systems

EXPERIENCE

DevOps Engineer — Capgemini (Indore)
January 2023 – Present (1.5 years)
- Maintained CI/CD pipelines using Jenkins and GitLab CI
- Managed Docker containers and Kubernetes cluster administration
- Monitored system health using Prometheus and Grafana
- No ML or data science involvement

EDUCATION
B.Tech Computer Science — IPS Academy, 2022 (CGPA: 6.8)

SKILLS
Docker (actual experience), Kubernetes (actual experience), Jenkins,
Python (scripting level), ML (self-study only), AWS (certifications)

PROJECTS
- CIFAR-10 image classifier (tutorial)
- Simple linear regression
- LangChain tutorial chatbot

CERTIFICATIONS
CKA (Kubernetes), AWS SysOps — 2023
""",
    },
    {
        "id": "candidate_040",
        "name": "Pooja Sharma",
        "ground_truth_quality": "low",
        "ground_truth_notes": "Fresh grad with no experience, inflated skills section with every possible ML keyword.",
        "candidate_type": "keyword_stuffer",
        "resume_text": """POOJA SHARMA
Senior ML Engineer | LLM Expert | MLOps | GenAI | Computer Vision
pooja.s.ai@gmail.com | Pune, India

PROFESSIONAL SUMMARY
Expert Machine Learning Engineer with deep knowledge of PyTorch, TensorFlow,
LLMs, RAG pipelines, MLOps (MLflow, Kubeflow, Airflow), Docker, Kubernetes,
AWS SageMaker, GCP Vertex AI, recommendation systems, NLP, computer vision,
fraud detection, time series, A/B testing, feature engineering, SHAP, LIME.

EXPERIENCE: None

EDUCATION
B.Tech Computer Science — Symbiosis Institute, 2024 (CGPA: 7.3)

SKILLS (claimed)
All of: Python, R, Java, Scala, SQL, NoSQL, PyTorch, TensorFlow, JAX,
scikit-learn, XGBoost, LightGBM, HuggingFace, LangChain, FAISS,
Docker, Kubernetes, Airflow, MLflow, AWS, GCP, Azure...

PROJECTS
- 10 tutorial-based projects from Coursera and Udemy
- No original work or deployed systems

CERTIFICATIONS
25 certificates from various online platforms
""",
    },
]

# ============================================================================
# GROUP 5: Career changers with transferable skills (10)
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_041",
        "name": "Ramesh Babu",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Former software engineer pivoting to ML, 6 years SE experience transferable, 1.5 years ML.",
        "candidate_type": "career_changer",
        "resume_text": """RAMESH BABU
Software Engineer transitioning to Machine Learning
ramesh.b@email.com | Bangalore, India

SUMMARY
6 years of senior software engineering at Oracle India, pivoting to ML engineering.
Built distributed systems serving 100M+ users. Now applying that engineering rigor
to ML systems. 1.5 years of hands-on ML work alongside my SE role.

EXPERIENCE

Senior Software Engineer — Oracle India (Bangalore)
July 2018 – Present (6 years)
- Built distributed microservices for Oracle Cloud Infrastructure
- Led migration of monolith to microservices (12 services, 2M LOC)
- ML side project: Built anomaly detection for Oracle cloud service SLAs
  using time-series LSTM — deployed to production, monitoring 500+ services

EDUCATION
B.Tech Computer Science — RVCE Bangalore, 2018 (CGPA: 8.6)

SKILLS
Python, Java, Kubernetes, Docker, distributed systems, REST APIs,
PyTorch (self-learning), scikit-learn, time-series, anomaly detection,
system design, SQL, microservices
""",
    },
    {
        "id": "candidate_042",
        "name": "Shalini Gupta",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "PhD physicist pivoting to ML, 5 years domain expertise in applied physics.",
        "candidate_type": "career_changer",
        "resume_text": """SHALINI GUPTA
PhD Physicist | ML for Science
shalini.g@email.com | Delhi, India

SUMMARY
PhD in computational physics from IIT Delhi, now applying ML to scientific problems.
Expert in numerical methods, statistical modeling, and high-performance computing.
Joined ML team at ISRO as data scientist 1.5 years ago.

EXPERIENCE

Data Scientist — ISRO (Bangalore)
January 2023 – Present (1.5 years)
- Applying ML to satellite image analysis for land classification
- Built neural network for cloud mask detection (accuracy: 92%)

Research Scholar — IIT Delhi
July 2018 – December 2022 (4.5 years)
- PhD: Applied ML to protein folding prediction for quantum chemistry models
- Published 3 papers in Physical Review Letters

EDUCATION
PhD Computational Physics — IIT Delhi, 2022
M.Sc. Physics — Delhi University, 2018

SKILLS
Python, NumPy, SciPy, PyTorch, scikit-learn, HPC (MPI, OpenMP),
MATLAB, statistical modeling, physics simulation, ML for science

PUBLICATIONS: 3 peer-reviewed papers in computational physics
""",
    },
    {
        "id": "candidate_043",
        "name": "Krishnaraj Menon",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Former financial analyst with 4 years quant finance, pivoting to ML, strong stats background.",
        "candidate_type": "career_changer",
        "resume_text": """KRISHNARAJ MENON
Quantitative Analyst | ML Engineer
krish.m@email.com | Mumbai, India

EXPERIENCE

Quantitative Analyst — HSBC India (Mumbai)
June 2019 – Present (5 years)
- Built statistical risk models for fixed income portfolio management
- Applied ML to equity factor model (improved Sharpe from 1.12 to 1.38)
- Python and R for data analysis, backtesting, and reporting
- Last 1.5 years: 50% time on ML project (credit scoring with XGBoost)

EDUCATION
CFA (Level 3 cleared), FRM
M.Sc. Statistics — CMI Chennai, 2019

SKILLS
Python, R, pandas, NumPy, scikit-learn, XGBoost, SQL, statistics,
time series, risk modeling, backtesting, financial markets, Docker (beginner)
""",
    },
    {
        "id": "candidate_044",
        "name": "Deepa Nair",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Doctor turned clinical AI researcher, domain expertise + 2 years ML.",
        "candidate_type": "career_changer",
        "resume_text": """DEEPA NAIR MBBS, MD
Clinical AI Researcher
deepa.n.ai@email.com | Pune, India

SUMMARY
MBBS + MD physician with 5 years clinical practice, now building AI for healthcare.
Domain expertise in clinical data interpretation. 2 years building medical ML models.

EXPERIENCE

Clinical AI Researcher — Manipal Hospitals AI Lab (Bangalore)
January 2022 – Present (2.5 years)
- Built AI model for sepsis prediction (AUROC: 0.87) using ICU vitals
- Designed clinical validation study with 2 hospitals (IRB approved)
- Presented at AMIA 2023 annual symposium

Medical Officer — KEM Hospital (Mumbai)
August 2017 – December 2021 (4.5 years)
- Clinical practice in emergency medicine

EDUCATION
MD Emergency Medicine — Mumbai University, 2017
MBBS — Government Medical College Pune, 2013

SKILLS
Python, scikit-learn, PyTorch, SQL, clinical data analysis,
survival analysis, EHR data processing, medical knowledge,
statistics, R, Pandas
""",
    },
    {
        "id": "candidate_045",
        "name": "Neeraj Pandey",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Ex-civil engineer, 4 years SE, pivoting to data science via Masters.",
        "candidate_type": "career_changer",
        "resume_text": """NEERAJ PANDEY
Data Scientist (ex-Civil Engineer)
neeraj.p@email.com | Varanasi, India

EXPERIENCE

Data Scientist — Delhivery (Gurgaon)
July 2023 – Present (1 year)
- Built warehouse location optimization model reducing last-mile cost by 11%
- Demand forecasting for 500+ delivery zones using XGBoost + SARIMA ensemble

Civil Engineer — L&T Construction (Mumbai)
June 2018 – June 2022 (4 years)
- Project management for infrastructure projects worth ₹200 crore+

EDUCATION
M.Tech Data Science — IIT Bombay, 2023
B.Tech Civil Engineering — IIT BHU, 2018

SKILLS
Python, pandas, scikit-learn, XGBoost, SQL, Airflow, logistics optimization,
time series, spatial analysis, geospatial Python (geopandas), NumPy
""",
    },
    {
        "id": "candidate_046",
        "name": "Sonali Bhatt",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Former UX researcher pivoting to ML for UX/behavior prediction, 3 years research.",
        "candidate_type": "career_changer",
        "resume_text": """SONALI BHATT
ML Engineer | User Behavior Analytics
sonali.b@email.com | Bangalore, India

EXPERIENCE

ML Engineer — Flipkart (Bangalore)
September 2022 – Present (2 years)
- Built user session analysis models predicting purchase intent (AUC: 0.82)
- Designed behavioral clustering of Flipkart shoppers into 8 segments;
  used by marketing team for targeted campaigns

UX Researcher — InMobi (Bangalore)
July 2019 – August 2022 (3 years)
- Led qualitative and quantitative user research for mobile advertising products
- Conducted A/B tests and analyzed user behavior data

EDUCATION
M.Des Human-Computer Interaction — IIT Bombay, 2019
B.Sc Psychology — Mumbai University, 2016

SKILLS
Python, scikit-learn, SQL, Tableau, user behavior analytics,
A/B testing, clustering, pandas, statistical testing, R
""",
    },
    {
        "id": "candidate_047",
        "name": "Manoj Verma",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Ex-network engineer pivoting to ML, 5 years network background + 2 years ML for networking.",
        "candidate_type": "career_changer",
        "resume_text": """MANOJ VERMA
Network Automation & ML Engineer
manoj.v@email.com | Chennai, India

EXPERIENCE

ML Engineer (Network AI) — Jio Platforms (Navi Mumbai)
March 2022 – Present (2.5 years)
- Built ML model predicting network congestion 15 minutes ahead (86% accuracy)
- Anomaly detection for network fault prediction, reducing MTTR by 30%

Network Engineer — Tata Communications (Chennai)
June 2017 – February 2022 (4.75 years)
- Managed enterprise WAN/LAN infrastructure for 200+ enterprise clients
- CCNP certified; led network migration projects

EDUCATION
B.Tech Electronics and Communication — VIT Vellore, 2017

SKILLS
Python, scikit-learn, PyTorch (learning), time series, anomaly detection,
Pandas, SQL, networking, Cisco, network protocols, Linux, Docker
""",
    },
    {
        "id": "candidate_048",
        "name": "Priti Singh",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Ex-biologist pivoting to ML/bioinformatics, strong domain expertise.",
        "candidate_type": "career_changer",
        "resume_text": """PRITI SINGH
Bioinformatics ML Researcher
priti.s@email.com | Hyderabad, India

EXPERIENCE

Bioinformatics Scientist — Biocon (Hyderabad)
January 2021 – Present (3.5 years)
- Built ML models for drug-target interaction prediction (AUC: 0.89)
- Applied graph neural networks to molecular property prediction
- Collaborated with computational chemistry team on ADMET prediction

Research Associate — CCMB Hyderabad
July 2018 – December 2020 (2.5 years)
- Genomics research on antimicrobial resistance

EDUCATION
M.Sc. Bioinformatics — University of Hyderabad, 2018
B.Sc. Biotechnology — JNTU, 2016

SKILLS
Python, PyTorch, scikit-learn, RDKit, BioPython, pandas, R,
graph neural networks, molecular biology, bioinformatics,
genomics, drug discovery ML, NumPy
""",
    },
    {
        "id": "candidate_049",
        "name": "Vijay Kumar",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Ex-teacher pivoting to data science, strong statistical thinking, 2 years DS.",
        "candidate_type": "career_changer",
        "resume_text": """VIJAY KUMAR
Data Scientist (ex-Mathematics Teacher)
vijay.k.ds@gmail.com | Bhopal, India

EXPERIENCE

Data Scientist — MP Government (Smart Cities Mission)
April 2022 – Present (2 years)
- Built traffic flow prediction model for Bhopal smart city initiative
- Citizen service request classification and routing (NLP, 91% accuracy)

Mathematics Teacher — Kendriya Vidyalaya
June 2016 – March 2022 (6 years)
- Taught mathematics and statistics to grades 9-12

EDUCATION
M.Sc. Mathematics — IIT Indore, 2016 (CGPA: 8.8)

SKILLS
Python, pandas, scikit-learn, SQL, statistics, NLP, matplotlib,
Streamlit, time series, spatial analysis, teaching, communication

STRENGTHS: Extremely strong mathematical foundation; excellent communicator
""",
    },
    {
        "id": "candidate_050",
        "name": "Anjali Rao",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "Ex-content strategist pivoting to NLP, 3 years domain + 1.5 years NLP.",
        "candidate_type": "career_changer",
        "resume_text": """ANJALI RAO
NLP Engineer (ex-Content Strategist)
anjali.rao@email.com | Bangalore, India

EXPERIENCE

NLP Engineer — Inshorts (Bangalore)
September 2022 – Present (2 years)
- Built auto-summarization system for news articles (ROUGE-L: 0.42)
- Topic modeling pipeline clustering 10,000+ articles daily

Content Strategist — Amazon India (Bangalore)
June 2019 – August 2022 (3.25 years)
- Led content strategy for Amazon India product pages
- Strong analytical skills; used SQL and Excel for content performance analysis

EDUCATION
MBA Marketing — XLRI Jamshedpur, 2019
B.A. English Literature — Lady Shri Ram College, 2017

SKILLS
Python, HuggingFace, scikit-learn, NLP, text summarization,
topic modeling, SQL, content strategy, communication, pandas
""",
    },
]

# ============================================================================
# GROUP 6: Domain specialists (10) — deep in one area, weak elsewhere
# ============================================================================

CANDIDATES += [
    {
        "id": "candidate_051",
        "name": "Dr. Rajan Krishnan",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "10 years NLP research, deep theoretical expertise, weak production/MLOps skills.",
        "candidate_type": "domain_specialist",
        "resume_text": """DR. RAJAN KRISHNAN
Senior Research Scientist — Natural Language Processing
rajan.k@iitm.ac.in | Chennai, India

SUMMARY
10 years of NLP research (IIT Madras, INRIA France). 35 publications,
h-index 18. Deep expertise in syntactic parsing, semantic representation,
and multilingual NLP for Indian languages. Limited production ML experience.

EXPERIENCE

Associate Professor (on sabbatical) — IIT Madras
2019 – Present (5 years)
- Leads the Language Technologies Research Centre
- Published at ACL, EMNLP, NAACL, COLING regularly

Research Scientist — INRIA Grenoble
2018 – 2019 (1 year postdoc)

PhD Research — IIT Bombay
2014 – 2018 (4 years)

EDUCATION
PhD Computer Science (NLP) — IIT Bombay, 2018
M.Tech — IIT Madras, 2014

SKILLS
Python, PyTorch, HuggingFace, Stanford NLP, NLTK, research,
syntactic parsing, semantic parsing, multilingual NLP, linguistics,
LaTeX, R (statistics), limited production/DevOps skills

PUBLICATIONS: 35, h-index 18 (Google Scholar)
""",
    },
    {
        "id": "candidate_052",
        "name": "Girish Sharma",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "8 years computer vision research, strong vision but no distributed systems or MLOps.",
        "candidate_type": "domain_specialist",
        "resume_text": """GIRISH SHARMA
Principal CV Researcher
girish.s@iisc.ac.in | Bangalore, India

SUMMARY
8 years computer vision research at IISc and TCS Research. 22 publications.
World-class expertise in 3D scene understanding and generative image models.
Transitioning to industry — limited production engineering experience.

EXPERIENCE

Principal Research Scientist — TCS Research (Bangalore)
2020 – Present (4 years)
- Research on 3D scene reconstruction and novel view synthesis (NeRF variants)
- 3 patents filed; work licensed to Tata Motors for autonomous driving

Postdoc — IISc Bangalore
2018 – 2020 (2 years)

PhD Computer Vision — IIT Bombay
2014 – 2018

EDUCATION
PhD Computer Science — IIT Bombay, 2018
B.Tech Electronics — BITS Pilani, 2014

SKILLS
Python, PyTorch, OpenCV, CUDA, 3D vision, generative models,
NeRF, image synthesis, weak in: Docker, Kubernetes, SQL, production systems

PUBLICATIONS: 22, including CVPR oral (2022)
""",
    },
    {
        "id": "candidate_053",
        "name": "Nalini Raghunathan",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "7 years biostatistics, deep statistical expertise but limited ML engineering.",
        "candidate_type": "domain_specialist",
        "resume_text": """NALINI RAGHUNATHAN
Senior Biostatistician / ML for Clinical Trials
nalini.r@email.com | Chennai, India

SUMMARY
7 years in biostatistics for clinical trials at pharma companies.
Expert in survival analysis, Bayesian adaptive trial design, mixed models.
Applying ML to clinical data; limited software engineering depth.

EXPERIENCE

Senior Biostatistician — Sun Pharmaceuticals (Mumbai)
2020 – Present (4 years)
- Bayesian adaptive trial design for Phase II oncology trials
- Applying ML to predict trial dropout and optimize enrollment
- R and Python for statistical programming

Biostatistician — Syngene International (Bangalore)
2017 – 2020 (3 years)

EDUCATION
M.Sc. Biostatistics — CMC Vellore, 2017
B.Sc. Statistics — Presidency College Chennai, 2015

SKILLS
R, Python, SAS, survival analysis, Bayesian methods, mixed models,
clinical trials, regulatory statistics, pandas, scikit-learn (basic),
limited: ML engineering, deployment, scalable systems
""",
    },
    {
        "id": "candidate_054",
        "name": "Suresh Naidu",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "8 years operations research, excellent optimization skills but weak ML.",
        "candidate_type": "domain_specialist",
        "resume_text": """SURESH NAIDU
Operations Research Scientist
suresh.n@email.com | Bangalore, India

SUMMARY
8 years in operations research and combinatorial optimization for supply chain.
Expert in LP, MIP, and heuristic methods. Applying ML to optimization.
Limited deep learning or ML platform experience.

EXPERIENCE

Senior Operations Research Scientist — Walmart Global Tech (Bangalore)
2020 – Present (4 years)
- Built mixed-integer programming model for Walmart US store replenishment
  optimization saving $50M annually
- Applied ML to forecast solver run times and warm-start solutions

Operations Research Analyst — IBM Global Services (Bangalore)
2016 – 2020 (4 years)

EDUCATION
M.Tech Industrial Engineering — IISc, 2016
B.Tech Mechanical — NITK Surathkal, 2014

SKILLS
Python, OR-Tools, Gurobi, CPLEX, scikit-learn (basic), pandas,
linear programming, integer programming, heuristics, logistics optimization,
supply chain, limited: PyTorch, deep learning, distributed systems
""",
    },
    {
        "id": "candidate_055",
        "name": "Preeti Bhojwani",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "6 years time series expert, deep ARIMA/state space expertise, limited DL.",
        "candidate_type": "domain_specialist",
        "resume_text": """PREETI BHOJWANI
Senior Data Scientist — Time Series Forecasting
preeti.b@email.com | Mumbai, India

SUMMARY
6 years specialized in time series forecasting for energy and financial markets.
Deep expertise in SARIMA, state space models, and Fourier analysis.
Transitioning to deep learning time series methods.

EXPERIENCE

Senior Data Scientist — BPCL (Mumbai)
2021 – Present (3 years)
- Fuel demand forecasting for 5,000+ petrol stations across India
- Reduced forecasting error (MAPE) from 8.2% to 4.1% using ensemble methods
- Built daily automated forecasting pipeline in Python + Airflow

Data Scientist — ICICI Direct (Mumbai)
2018 – 2021 (3 years)
- Stock price and volatility forecasting models

EDUCATION
M.Sc. Statistics (Time Series specialization) — IIT Kanpur, 2018

SKILLS
Python, R, statsmodels, ARIMA, SARIMA, state space models,
Prophet, scikit-learn, Airflow, SQL, pandas, NumPy,
limited: PyTorch, deep learning, ML deployment at scale
""",
    },
    {
        "id": "candidate_056",
        "name": "Amitabh Dasgupta",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "9 years reinforcement learning specialist, academic-industry mix.",
        "candidate_type": "domain_specialist",
        "resume_text": """AMITABH DASGUPTA
Reinforcement Learning Specialist
amitabh.d@email.com | Kolkata, India

SUMMARY
9 years focused exclusively on reinforcement learning and simulation.
Published 8 papers on RL for robotics and game-playing agents.
Recently moved to industry; limited production ML experience.

EXPERIENCE

Senior ML Scientist — Google DeepMind India (Hyderabad)
2022 – Present (2 years)
- Research on multi-agent RL for resource allocation

Research Scientist — IIT Kharagpur
2015 – 2022 (7 years postdoc + faculty)

EDUCATION
PhD Computer Science (RL) — IIT Bombay, 2015

SKILLS
Python, PyTorch, JAX, RLlib, OpenAI Gym, reinforcement learning,
multi-agent systems, simulation, game theory,
limited: MLOps, SQL, distributed data systems, production deployment

PUBLICATIONS: 8, including NeurIPS (2021, 2023)
""",
    },
    {
        "id": "candidate_057",
        "name": "Shweta Borkar",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "7 years geospatial ML expert, deep domain but niche.",
        "candidate_type": "domain_specialist",
        "resume_text": """SHWETA BORKAR
Geospatial ML Scientist
shweta.b@email.com | Pune, India

SUMMARY
7 years in geospatial data science and remote sensing ML.
Expert in satellite imagery analysis, GIS, and spatial statistics.
Limited experience with general ML platform engineering.

EXPERIENCE

Senior Geospatial Scientist — Esri India (Pune)
2019 – Present (5 years)
- Built crop yield prediction model using Sentinel-2 satellite data (RMSE: 0.18 t/ha)
- Flood extent mapping model for National Disaster Management Authority

GIS Analyst — Survey of India (Dehradun)
2017 – 2019 (2 years)

EDUCATION
M.Tech Remote Sensing — IIT Roorkee, 2017
B.Tech Civil Engineering — VNIT Nagpur, 2015

SKILLS
Python, PyTorch, rasterio, GDAL, geopandas, scikit-learn,
remote sensing, GIS, spatial statistics, object detection,
satellite imagery, limited: MLOps, distributed systems, SQL at scale
""",
    },
    {
        "id": "candidate_058",
        "name": "Akshay Pandita",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "8 years speech recognition expert, strong signal processing.",
        "candidate_type": "domain_specialist",
        "resume_text": """AKSHAY PANDITA
Senior Speech ML Engineer
akshay.p@email.com | Mumbai, India

SUMMARY
8 years in automatic speech recognition and text-to-speech for Indian languages.
Led Hindi and Kashmiri ASR development at CDAC. Expert in acoustic modeling
and speech signal processing. Limited general ML engineering experience.

EXPERIENCE

Senior Research Engineer — CDAC Mumbai
2016 – Present (8 years)
- Led Hindi ASR project (WER: 8.2% on AIR broadcast corpus)
- Built 5 Indian language TTS systems deployed in government digital services
- Kaldi, ESPnet, and custom PyTorch models

EDUCATION
M.Tech Speech Technology — IIT Madras, 2016
B.Tech ECE — NIT Kashmir, 2014

SKILLS
Python, PyTorch, Kaldi, ESPnet, librosa, torchaudio,
speech recognition, TTS, acoustic modeling, DSP,
limited: Docker, Kubernetes, web deployment, SQL
""",
    },
    {
        "id": "candidate_059",
        "name": "Vandana Iyer",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "6 years graph ML specialist, strong knowledge graphs and network analysis.",
        "candidate_type": "domain_specialist",
        "resume_text": """VANDANA IYER
Graph ML Research Engineer
vandana.i@email.com | Bangalore, India

SUMMARY
6 years in graph machine learning and knowledge graphs. Built knowledge
graph systems at Wipro Research used by 3 enterprise clients. Deep in
graph neural networks and entity resolution. Limited production MLOps.

EXPERIENCE

Research Engineer — Wipro Research (Bangalore)
2020 – Present (4 years)
- Built enterprise knowledge graph for a US telecom client (50M+ entities)
- Graph neural network for entity resolution (99.1% precision)
- Fraud ring detection using GNN on transaction graphs

Data Scientist — EY GDS (Bangalore)
2018 – 2020 (2 years)
- Built customer analytics models for financial services clients

EDUCATION
M.Tech Computer Science — IIT Delhi, 2018

SKILLS
Python, PyTorch Geometric, DGL, NetworkX, Neo4j, scikit-learn,
graph neural networks, knowledge graphs, entity resolution,
community detection, SQL (graph extensions), limited: distributed training, MLOps
""",
    },
    {
        "id": "candidate_060",
        "name": "Harish Subramanian",
        "ground_truth_quality": "medium",
        "ground_truth_notes": "10 years Bayesian ML specialist, probabilistic programming expert.",
        "candidate_type": "domain_specialist",
        "resume_text": """HARISH SUBRAMANIAN
Principal Bayesian ML Scientist
harish.s@email.com | Chennai, India

SUMMARY
10 years in probabilistic machine learning and Bayesian methods.
Deep expertise in variational inference, Gaussian processes, and
uncertainty quantification. Academic-industry hybrid career.
Limited modern MLOps and deployment experience.

EXPERIENCE

Principal Scientist — Tata Consultancy Services Research (Chennai)
2019 – Present (5 years)
- Built Bayesian demand forecasting system with calibrated uncertainty bands
  adopted by 2 retail clients
- Uncertainty quantification for medical AI systems meeting FDA guidance

Research Scientist — INRIA Paris (France)
2014 – 2018 (4 years postdoc)

EDUCATION
PhD Statistics (Bayesian Methods) — IIT Madras, 2014
M.Stat — ISI Kolkata, 2010

SKILLS
Python, PyMC, Stan, NumPyro, GPyTorch, scikit-learn, R, pandas,
Bayesian methods, Gaussian processes, variational inference,
probabilistic programming, uncertainty quantification,
limited: PyTorch DL, Kubernetes, CI/CD, production ML systems

PUBLICATIONS: 15, h-index 12
""",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Write files
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert len(CANDIDATES) == 60, f"Expected 60 candidates, got {len(CANDIDATES)}"

    for i, candidate in enumerate(CANDIDATES, 1):
        filename = f"candidate_{i:03d}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Ensure id matches filename
        candidate["id"] = f"candidate_{i:03d}"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(candidate, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Generated {len(CANDIDATES)} synthetic candidate profiles in:")
    print(f"   {OUTPUT_DIR}")

    # Print group summary
    from collections import Counter
    types = Counter(c["candidate_type"] for c in CANDIDATES)
    qualities = Counter(c["ground_truth_quality"] for c in CANDIDATES)

    print("\nCandidate type breakdown:")
    for t, count in sorted(types.items()):
        print(f"  {t:25s}: {count}")

    print("\nGround truth quality breakdown:")
    for q, count in sorted(qualities.items()):
        print(f"  {q:10s}: {count}")

    print("\nSpecial candidates:")
    for c in CANDIDATES:
        if c["name"] in ["Arjun Sharma", "Ravi Mehta"]:
            print(f"  [*] {c['name']}: {c['id']} ({c['candidate_type']}, {c['ground_truth_quality']})") 


if __name__ == "__main__":
    main()
