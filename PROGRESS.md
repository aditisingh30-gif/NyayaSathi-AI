# NyayaSathi (न्यायसाथी) — Progress Log

> AI Legal & Policy Assistant · React + Vite · FastAPI · ChromaDB · Gemini API (RAG)

---

## ✅ Project Status: FULLY OPERATIONAL

**Last updated**: 2026-11-11  
**Backend**: http://localhost:8000  
**Frontend**: http://localhost:5173

---

## 🏗️ Architecture

| Layer | Technology | Status |
|---|---|---|
| Frontend | React 18 + Vite + TailwindCSS | ✅ Running |
| Backend | FastAPI + Uvicorn (port 8000) | ✅ Running |
| Vector Store | ChromaDB 1.5.9 (persistent, cosine similarity) | ✅ Indexed |
| LLM | Google Gemini (gemini-2.0-flash) with offline fallback | ✅ Fallback active |
| PDF Processing | PyMuPDF (fitz) | ✅ Working |
| Retrieval | Hybrid lexical + semantic (RAG) | ✅ Fixed & verified |

---

## 📚 Indexed Legal Corpus (Auto-indexed on startup)

| Document | Category | Chunks |
|---|---|---|
| Digital Personal Data Protection Act, 2023 | Data Protection | ~4 |
| Indian Contract Act, 1872 (Essential Provisions) | Contract Law | ~4 |
| Bharatiya Nyaya Sanhita (BNS), 2023 | Criminal Law | ~4 |
| Standard Enterprise Cloud & AI MSA | Commercial Contract | ~2 |

**Total**: 4 documents · 14 chunks

---

## 🔧 Features Implemented

### Core Features
- [x] **Legal Query Assistant** — RAG-grounded Q&A with citations from indexed statutes
- [x] **Conversational Chat** — Multi-turn legal discussions with context retention
- [x] **Document Library** — Upload, index, and manage PDF legal documents
- [x] **Contract Clause Analyzer** — Risk flagging for legal clauses (CRITICAL/HIGH/MEDIUM/LOW)
- [x] **Policy Compliance Auditor** — DPDP Act compliance gap analysis
- [x] **Gemini API Key Management** — Runtime key configuration via UI

### RAG Pipeline
- [x] **Query Expansion** — Enriches queries with formal legal terminology
- [x] **Hybrid Retrieval** — 55% lexical + 45% vector similarity scoring
- [x] **Stop-word Filtering** — Removes common words from embedding generation
- [x] **Bigram Support** — Captures legal phrases (e.g., "data_breach", "personal_data")
- [x] **Zero-overlap Guardrail** — Rejects chunks with no substantive keyword overlap
- [x] **Exact Article-Number Metadata Matching** — Fast, prioritized direct index lookup for articles (e.g. "21", "21a")
- [x] **Section Mismatch Guard** — Prevents "Article 21" from matching "Section 124" or "Article 21A"
- [x] **Schedule / Appendix Filter** — Prevents enumeration chunks from Schedules (p299, p313) from polluting article queries
- [x] **Source Unavailable Message** — Clear disclosure when KB lacks the queried topic

---

## 🗄️ ChromaDB Safe Re-Indexing (Stage 3 Complete)

- **Backup Location**: `backend/data/chromadb_backup_stage3`
- **Target Document ID**: `doc_223b42d0c4` (Constitution of India)
- **Re-indexed Chunks**: 1,311 chunks (extracted using corrected article boundary detection)
- **Total Chunks in Corpus**: 1,325 chunks across 5 documents
- **Preserved Documents**:
  1. Digital Personal Data Protection Act, 2023 (6 chunks)
  2. Indian Contract Act, 1872 (Essential Provisions) (3 chunks)
  3. Bharatiya Nyaya Sanhita (BNS) 2023 - Commercial & Cyber Offences (2 chunks)
  4. Standard Enterprise Cloud & AI Master Services Agreement (3 chunks)
  5. Constitution of India (`doc_223b42d0c4`) (1,311 chunks)

---

## 🐛 Bug Fix: RAG Retrieval Accuracy (Critical)

### Problem
When querying "What is Article 21 of the Constitution of India?", the system was
returning and citing unrelated provisions:
- Section 124 — Indian Contract Act (Indemnity)
- Section 9 — DPDP Act (Children's Data)

**Root cause**: Pure cosine similarity with no lexical grounding was permissive enough
to surface semantically-adjacent but legally-unrelated chunks.

### Fix Applied

**1. `backend/app/services/vector_service.py`** — Hybrid scoring + guardrails
- Added `extract_substantive_tokens()`: strips stop words, extracts substantive legal terms
- `LegalSemanticEmbeddingFunction`: stop-word-filtered embeddings with bigram support
- **Guardrail 1**: Zero substantive token overlap → hard reject (not threshold-based)
- **Guardrail 2**: Explicit section number mismatch → reject
- **Bonus**: Exact target section in chunk header → +0.25 score boost
- `min_relevance_threshold=0.35` applied after hybrid scoring
- Added `original_query` parameter: prevents query-expansion section numbers from triggering false mismatch guards

**2. `backend/app/services/rag_service.py`** — Empty-results handling
- Passes `min_relevance_threshold=0.35` explicitly
- Passes `original_query=request.question` to prevent expansion artifacts
- Returns detailed "Source Not Available" message with zero citations when no chunks pass guardrails

**3. `backend/app/services/gemini_service.py`** — Consistent empty-state message
- Updated fallback message to match the same "source unavailable" language

### Verification

**Before fix:**
```
Query: "What is Article 21 of the Constitution of India?"
Result: 3 citations — Section 124 ICA, Section 33 DPDP, Section 9 DPDP ❌
```

**After fix:**
```
Query: "What is Article 21 of the Constitution of India?"
Result: citations=[], confidence=0.0, answer="Source Not Available..." ✅

Query: "What is the penalty for data breach under the DPDP Act?"
Result: 3 citations from DPDP Act, confidence=0.697 ✅
```

---

## 🧪 Test Results

### Unit Tests (via `python backend/run_tests.py`)

| Test File | Tests | Status |
|---|---|---|
| `test_pdf_service.py` | 1 | ✅ PASSED |
| `test_vector_service.py` | 1 | ✅ PASSED |
| `test_rag_service.py` | 1 | ✅ PASSED |
| `test_retrieval_accuracy.py` | 10 | ✅ ALL PASSED |
| **Total** | **13** | **✅ 13/13** |

### Retrieval Accuracy Tests (`test_retrieval_accuracy.py`)

| Test | Assertion | Result |
|---|---|---|
| `test_dpdp_data_breach_returns_dpdp_results` | DPDP query → DPDP citations | ✅ PASS |
| `test_article_21_returns_no_results` | Constitution query → 0 results | ✅ PASS |
| `test_section_420_ipc_returns_no_results` | IPC query → 0 results | ✅ PASS |
| `test_section_73_contract_act_returns_correct_doc` | Sec 73 query → Contract Act | ✅ PASS |
| `test_section_73_does_not_return_section_124` | Sec 73 query → no Sec 124 | ✅ PASS |
| `test_rag_returns_source_unavailable_for_constitution_query` | Full RAG → no citations, source unavailable message | ✅ PASS |
| `test_rag_returns_citations_for_dpdp_query` | DPDP RAG → ≥1 citation | ✅ PASS |
| `test_relevance_scores_are_valid` | Scores in [0, 1] | ✅ PASS |
| `test_results_are_sorted_by_relevance` | Descending sort | ✅ PASS |
| `test_bns_cheating_section_returned_for_cheating_query` | BNS cheating chunk returned | ✅ PASS |

### Stage 1 & 2 RAG Precision Tests (`test_article_boundaries.py` & `test_stage2_retrieval.py`)

| Test | Suite | Assertion | Result |
|---|---|---|---|
| `test_article_boundaries_page_42` | `test_article_boundaries.py` | Page 42 split into Art 20, 21, 21A, 22 | ✅ PASS |
| `test_article_21_exact_retrieval` | `test_stage2_retrieval.py` | Exact article_number="21" prioritized on page 42 | ✅ PASS |
| `test_article_21a_not_confused_with_21` | `test_stage2_retrieval.py` | Art 21A queries retrieve 21A, not 21 | ✅ PASS |
| `test_article_22_retrieval` | `test_stage2_retrieval.py` | Art 22 queries retrieve Art 22 | ✅ PASS |
| `test_schedules_filtered_out_for_article_queries` | `test_stage2_retrieval.py` | Schedule pages (p299, p313) filtered out | ✅ PASS |
| `test_missing_article_returns_empty` | `test_stage2_retrieval.py` | Missing article yields 0 results | ✅ PASS |
| `test_rag_service_missing_article_no_citations` | `test_stage2_retrieval.py` | RAG yields no citations & unavailable notice | ✅ PASS |
| `test_rag_service_article_21_returns_correct_citation` | `test_stage2_retrieval.py` | RAG yields citation with page 42 & Art 21 text | ✅ PASS |

### Integration Tests (`python backend/tests/run_api_tests.py`)

| Test | Endpoint | Result |
|---|---|---|
| Root Endpoint | `GET /` | ✅ PASS |
| Health Check | `GET /api/health` | ✅ PASS |
| Documents Listing | `GET /api/documents` | ✅ PASS |
| RAG Legal Query | `POST /api/query` | ✅ PASS |
| Conversational Chat | `POST /api/query/chat` | ✅ PASS |
| Clause Risk Analysis | `POST /api/analysis/clause` | ✅ PASS |
| Policy Compliance Audit | `POST /api/analysis/policy` | ✅ PASS |
| Document Lifecycle | Upload → Query → Delete | ✅ PASS |
| **Total** | | **✅ 8/8** |

---

## 📁 Project Structure

```
ai_pro/
├── backend/
│   ├── app/
│   │   ├── config.py               # Settings & paths
│   │   ├── main.py                 # FastAPI app + auto-indexer
│   │   ├── models/schemas.py       # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── documents.py        # Upload, list, delete
│   │   │   ├── query.py            # RAG query + chat
│   │   │   ├── analysis.py         # Clause & policy analysis
│   │   │   └── health.py           # Health check
│   │   ├── services/
│   │   │   ├── vector_service.py   # ChromaDB + hybrid retrieval [FIXED]
│   │   │   ├── rag_service.py      # RAG orchestration [FIXED]
│   │   │   ├── gemini_service.py   # LLM + fallback engine [UPDATED]
│   │   │   ├── pdf_service.py      # PyMuPDF extraction
│   │   │   └── sample_data_loader.py # Benchmark corpus loader
│   │   └── sample_data/            # 4 benchmark PDF statutes
│   ├── tests/
│   │   ├── test_pdf_service.py
│   │   ├── test_vector_service.py
│   │   ├── test_rag_service.py
│   │   ├── test_retrieval_accuracy.py  # NEW — 10 accuracy tests
│   │   └── run_api_tests.py
│   ├── run.py                      # Uvicorn launcher
│   └── run_tests.py                # Pytest wrapper
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── Navbar.jsx
│       │   ├── ChatInterface.jsx
│       │   ├── DocumentManager.jsx
│       │   ├── ClauseAnalyzer.jsx
│       │   ├── PolicyAuditor.jsx
│       │   └── ApiKeyModal.jsx
│       └── services/api.js
├── PROGRESS.md                     # This file
└── README.md
```

---

## 🚀 How to Run

```powershell
# Terminal 1 — Backend
python backend/run.py

# Terminal 2 — Frontend
cd frontend; npm run dev -- --host --port 5173

# Run all tests
python backend/run_tests.py backend/tests/test_retrieval_accuracy.py backend/tests/test_rag_service.py backend/tests/test_vector_service.py backend/tests/test_pdf_service.py

# Integration tests
python backend/tests/run_api_tests.py
```

---

## ⚠️ Known Limitations

1. **Constitution of India** is not in the knowledge base. Upload it via Document Library to query fundamental rights.
2. **Gemini API** requires a valid key set via the UI or `.env` file. Without it, the offline fallback generates template responses.
3. The offline fallback synthesizer may reference the Enterprise MSA for queries about "data breach" since it's also indexed. This is correct (the MSA contains data breach clauses) but may be unexpected.

---

*NyayaSathi never fabricates citations. Only verified provisions from indexed documents are presented.*
