💼 Deloitte-Ready Twitter Sentiment Intelligence

A production-grade sentiment analysis accelerator designed to showcase the end-to-end delivery skills expected from a Deloitte India Oracle Analyst.
The solution spans data ingestion, preprocessing, model training, governance, deployment readiness, and business storytelling — all within a lightweight, reproducible ML pipeline.

🧰 Technology Stack & Analytics Capabilities

Python, Pandas, Scikit-learn, NLTK – rapid experimentation & production-ready sentiment pipelines

TF-IDF Vectorization + Multiclass Logistic Regression – explainable predictions across positive, neutral, negative classes

Streamlit UI (app/app.py) – displays class probabilities & narrative insights for stakeholders

Notebook-driven exploration (notebooks/) – retraining lineage and model interpretability

Container-ready CI/CD pipeline – automates tests, linting, and retraining

🧭 Why This Project Stands Out

Oracle ecosystem alignment – optional ingestion from Oracle Autonomous Database with OCI deployment playbooks

Consulting-grade engineering – modular Python package, logging, configuration management, GitHub Actions CI

Business storytelling – dashboards, KPIs, and executive-friendly insights translating data into measurable ROI

🔗 Live Resources & Quick Links
Asset	Purpose	Link
Live Streamlit Demo	Interview-ready interactive dashboard	https://<your-app>.streamlit.app

Architecture Blueprint	Deep dive on data, model & DevOps layers	docs/architecture.md

OCI Deployment Playbook	Containerized rollout on Oracle Cloud	deployment/oracle_cloud.md

Streamlit Cloud Guide	Launch public URL in minutes	deployment/streamlit_cloud.md

Vercel Redirect Playbook	Vanity domain / QR-friendly link	deployment/vercel_redirect.md

GitHub Live Validation	Verify live deployment health	deployment/github_live_validation.md

Training Data Sample	Quick-start dataset for retraining	data/twitter_training.csv

✅ After publishing, replace the placeholder URL above and embed it directly in your resume / interview slides.

🏗️ Architecture Overview
flowchart LR
    subgraph Data_Layer
        A[Oracle Autonomous DB] -- optional --> B[(CSV Data)]
    end
    subgraph Processing_Layer
        B --> C[Data Loader]
        C --> D[Text Preprocessor]
        D --> E[Scikit-learn Pipeline]
        E --> F[Model Metrics]
    end
    subgraph Experience_Layer
        E --> G[Streamlit Dashboard]
        E --> H[CLI Automation]
        F --> G
        F --> I[Reporting / GitHub Pages]
    end
    subgraph DevOps_Layer
        J[GitHub Actions CI]
        J --> E
        J --> G
    end

📁 Repository Structure
Path	Description
app/	Streamlit app for demos
artifacts/	Generated model artifacts (gitignored; created via scripts/train.py)
config/	Central YAML settings controlling ingestion & deployment toggles
data/	Sample labelled tweets
deployment/	Multi-cloud deployment guides
docs/	Architecture diagrams & KPI catalogue
scripts/	Automation scripts for training & inference
src/twitter_sentiment/	Core reusable Python package
tests/	Pytest-based unit tests
.github/workflows/	CI pipeline (lint + tests)
🚀 Quick Start
# 1️⃣  Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2️⃣  Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3️⃣  Train pipeline & generate artifacts
python scripts/train.py

# 4️⃣  Launch Deloitte Storytelling Dashboard
streamlit run app/app.py


Note: Artifacts are excluded from source control.
Run python scripts/train.py whenever cloning the repo to recreate artifacts/sentiment_pipeline.joblib.

🧪 Quality Gates
Check	Command
Unit tests	pytest -q
Package compile check	python -m compileall src
Model retrain	python scripts/train.py

CI runs automatically on every push or PR to main.

🧩 Feature Highlights

Config-driven ingestion switch (CSV ⇄ Oracle DB)

Reusable text-cleaning module (URL, mention, punctuation handling)

Auditable Scikit-learn pipeline with persisted metrics

Streamlit dashboard → probabilities + governance tab

CLI utilities for automated workflows

Pytest suite + GitHub Actions CI/CD

☁️ Cloud Deployment

See deployment/oracle_cloud.md
 for the Oracle Cloud Infrastructure (OCI) guide:

Containerize Streamlit app using OCI Container Instances

Automate retraining via OCI Data Science jobs / GitHub Actions

Connect to Autonomous DB with wallet credentials

Or follow deployment/streamlit_cloud.md
 for a free Streamlit Cloud public link—perfect for interviews.

🗃️ Oracle Database Integration (Optional)
oracle_integration:
  enabled: true
  wallet_location: /path/to/wallet
  user: DATA_ENGINEER
  dsn: myadb_high
  sql_query: |
    SELECT text, sentiment
    FROM analytics.twitter_training_data
    WHERE created_at >= SYSDATE - 30

📊 Storytelling in Interviews

Business Impact – sentiment tracking → improved ROI & retention

Oracle Expertise – wallet connectivity, SQL modeling, OCI integration

Engineering Rigor – modular design, testing, CI/CD

Consulting Mindset – executive-ready Streamlit deliverable

📦 GitHub Best Practices

Use feature branches (e.g., feature/oracle-ingestion)

Raise PRs with CI status + screenshots

Track backlog via GitHub Projects

Tag releases (e.g., v1.0.0) post-deployment

🧠 Training Artifacts

Generated after running scripts/train.py:

sentiment_pipeline.joblib – Full TF-IDF + Logistic Regression pipeline

Legacy (compatible) files:

app/logistic_model.pkl

app/tfidf_vectorizer.pkl

sentiment_model.pkl

Use these for quick demos or backward comparisons.

🤝 Contributing

Pull requests welcome!
Please open an issue describing any enhancement or bugfix before major changes.
 
