# NyayaSathi (न्यायसाथी) — AI Legal & Policy Assistant

> Authoritative Legal Assistant grounded in Indian Statutory Law, Constitutional Jurisprudence, and Commercial Contracts. Powered by FastAPI, React + Vite, ChromaDB, and Google Gemini.

---

## 🏛️ Project Overview

**NyayaSathi** provides deterministic, evidence-grounded answers to statutory and contractual queries. It implements strict section-aware RAG guardrails:
- **Exact Article/Section Matching**: Dedicated metadata matching for statutory provisions (e.g., Article 21, Article 21A, Section 73).
- **Zero Hallucination Guarantee**: If an article or section is not in the indexed knowledge base, NyayaSathi explicitly states that the source is unavailable rather than citing unrelated provisions.
- **Contract Risk Analysis**: Built-in clause risk categorization (CRITICAL, HIGH, MEDIUM, LOW) for liability caps, indemnities, and termination clauses.
- **Policy Compliance Auditor**: Comprehensive compliance checklist against the Digital Personal Data Protection (DPDP) Act 2023.

---

## 👥 GitHub Collaboration Guidelines (4-Person Team)

### Roles & Access
- **Repository Owner**: Full administrative control and deployment management.
- **Teammates (3 Collaborators)**: Granted **Write** access (can create feature branches, commit, push, and open Pull Requests).

### Branching & PR Strategy
1. **Never commit directly to `main`**. The `main` branch is protected and holds production-ready code.
2. **Branch Naming**:
   - `feature/<name>-<topic>` (e.g., `feature/rohit-citation-formatting`, `feature/priya-pdf-highlighter`)
   - `fix/<name>-<bug>` (e.g., `fix/aman-clause-risk-threshold`)
3. **Submitting Changes**:
   ```bash
   git checkout -b feature/yourname-feature-description
   # Make your changes
   git add .
   git commit -m "feat: descriptive message of your work"
   git push origin feature/yourname-feature-description
   ```
4. Open a **Pull Request (PR)** against `main`. Require at least 1 teammate review before merging.

---

## 🔒 Security & Environment Rules

1. **NEVER commit `.env` or API keys**. All secrets and personal machine paths are excluded via `.gitignore`.
2. **API Keys belong ONLY on the backend**. The frontend must never have direct access to `GEMINI_API_KEY`.
3. Use the placeholder template `.env.example` to set up your local development environment.

---

## 💻 Local Setup Instructions for Teammates

### Prerequisites
- **Python 3.10+** (tested up to Python 3.14)
- **Node.js 18+** & `npm`
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/aditisingh30-gif/demo1.git nyayasathi
cd nyayasathi
```

### 2. Backend Setup (FastAPI)
```bash
# Optional: create and activate a virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Open .env and add your GEMINI_API_KEY if testing live LLM synthesis
# (If left empty, NyayaSathi runs automatically in local deterministic offline fallback mode!)

# Run backend
python backend/run.py
```
*Backend runs on `http://localhost:8000` (API Docs at `http://localhost:8000/docs`).*

### 3. Frontend Setup (React + Vite)
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
*Frontend runs on `http://localhost:5173`.*

### 4. Running Automated Tests
```bash
# Run complete test suite (25 test cases covering PDF boundaries, RAG accuracy, synthesis)
python backend/run_tests.py backend/tests/test_article_boundaries.py backend/tests/test_pdf_service.py backend/tests/test_vector_service.py backend/tests/test_rag_service.py backend/tests/test_retrieval_accuracy.py backend/tests/test_stage2_retrieval.py backend/tests/test_article_21_synthesis.py

# Run API integration tests
python backend/tests/run_api_tests.py
```

---

## ☁️ Online Deployment Guide

### A. Frontend Deployment (Vercel)
1. Import repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. Set **Framework Preset** to `Vite`.
4. Add Environment Variable:
   - `VITE_API_BASE_URL`: `https://your-backend-app.onrender.com`
5. Deploy. `vercel.json` will automatically route SPA requests.

### B. Backend Deployment (Render)
1. Create a new **Web Service** on [Render](https://render.com).
2. Root Directory: `.` (workspace root).
3. Runtime: `Python 3`.
4. Build Command: `pip install -r backend/requirements.txt`
5. Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variables:
   - `GEMINI_API_KEY`: *(Your Google AI Studio Key)*
   - `GEMINI_MODEL`: `gemini-1.5-flash`
   - `DEBUG`: `False`

### C. Persistent Storage & Legal Corpus on Cloud Deployment
> [!IMPORTANT]
> **ChromaDB Cloud Storage Strategy**:
> - Free/ephemeral server instances (like Render free tier) clear local disk storage when spinning down.
> - **Out-of-the-box Preload**: On fresh launch, the backend auto-indexes the 4 benchmark Indian statutes and contracts (`Digital Personal Data Protection Act 2023`, `Indian Contract Act 1872`, `Bharatiya Nyaya Sanhita 2023`, and `Enterprise Cloud AI MSA`).
> - **Adding Constitution / Large PDFs in Cloud**:
>   1. **Web UI Upload**: You or your teammates can upload any Act or Constitution PDF anytime via the **Document Library** tab; the system chunks and indexes it instantly in memory and disk.
>   2. **Persistent Disk (Production)**: Attach a **Render Persistent Disk** mounted at `/opt/render/project/src/backend/data` so uploaded PDFs and vector embeddings survive server restarts without re-uploading.

---

## 📂 Project Architecture

```
nyayasathi/
├── .env.example               # Safe environment variable template
├── .gitignore                 # Comprehensive security and build exclusion rules
├── README.md                  # Team setup and deployment guide
├── PROGRESS.md                # Task tracking & verification logs
├── backend/
│   ├── app/
│   │   ├── config.py          # Centralized configuration (relative paths)
│   │   ├── main.py            # FastAPI lifespan, CORS, route registration
│   │   ├── models/schemas.py  # Pydantic data schemas
│   │   ├── routers/           # Query, Documents, Analysis, Health
│   │   ├── services/
│   │   │   ├── pdf_service.py     # PyMuPDF parser with article boundary detection
│   │   │   ├── vector_service.py  # ChromaDB hybrid search & exact metadata matching
│   │   │   ├── rag_service.py     # RAG pipeline orchestration
│   │   │   ├── gemini_service.py  # Gemini LLM & deterministic evidence synthesizer
│   │   │   └── sample_data_loader.py # Benchmark statutes loader
│   │   └── sample_data/       # Built-in sample PDF legal statutes
│   ├── tests/                 # 25+ automated regression & integration tests
│   ├── requirements.txt       # Python dependencies
│   └── run.py                 # Backend runner
└── frontend/
    ├── vercel.json            # Vercel SPA rewrites
    ├── package.json           # React 18, Vite, Tailwind CSS, Lucide Icons
    ├── vite.config.js         # Vite configuration
    └── src/
        ├── App.jsx            # Multi-tab layout
        ├── services/api.js    # API client with VITE_API_BASE_URL support
        └── components/        # ChatInterface, DocumentManager, ClauseAnalyzer, etc.
```
