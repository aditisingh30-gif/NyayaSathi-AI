"""
test_retrieval_accuracy.py
==========================
Automated tests for RAG retrieval accuracy in NyayaSathi.

Tests verify:
1. Relevant queries return matching results from the correct document.
2. Irrelevant queries (e.g. topics not in KB) return ZERO results.
3. Section/article number mismatch is correctly rejected.
4. The full RAG pipeline returns "source unavailable" when KB lacks the topic.
5. Query expansion does not introduce false positives.

Run with:
    python backend/run_tests.py backend/tests/test_retrieval_accuracy.py
"""

import sys
import os

# Ensure backend is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
from app.services.vector_service import ChromaVectorService, LegalSemanticEmbeddingFunction
from app.services.rag_service import RAGLegalService
from app.services.gemini_service import GeminiLegalService
from app.models.schemas import QueryRequest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Use an in-memory / temp ChromaDB populated with known test chunks
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    # DPDP Act chunks
    {
        "chunk_id": "test_dpdp_1",
        "text": (
            "Section 8 of the Digital Personal Data Protection Act, 2023 imposes obligations on Data Fiduciaries "
            "to implement appropriate technical and organisational measures to protect personal data from breach. "
            "Data Fiduciaries must notify the Board and affected Data Principals of a personal data breach."
        ),
        "section_header": "Section 8 - Obligations of Data Fiduciary",
        "page_number": 5,
        "document_title": "Digital Personal Data Protection Act, 2023",
        "category": "Data Protection",
    },
    {
        "chunk_id": "test_dpdp_2",
        "text": (
            "Section 33 of the Digital Personal Data Protection Act, 2023 provides for financial penalties. "
            "The Data Protection Board may impose a monetary penalty of up to INR 250 crore for failure to "
            "implement adequate security safeguards."
        ),
        "section_header": "Section 33 - Penalties",
        "page_number": 18,
        "document_title": "Digital Personal Data Protection Act, 2023",
        "category": "Data Protection",
    },
    # Indian Contract Act chunks
    {
        "chunk_id": "test_ica_73",
        "text": (
            "Section 73 of the Indian Contract Act, 1872 states that when a contract has been broken, "
            "the party who suffers by such breach is entitled to receive compensation for any loss or damage "
            "caused to him thereby. The compensation is not to be given for any remote or indirect loss."
        ),
        "section_header": "Section 73 - Compensation for loss or damage caused by breach of contract",
        "page_number": 42,
        "document_title": "Indian Contract Act, 1872",
        "category": "Contract Law",
    },
    {
        "chunk_id": "test_ica_124",
        "text": (
            "Section 124 of the Indian Contract Act, 1872 defines a contract of indemnity as a contract "
            "by which one party promises to save the other from loss caused to him by the conduct of the "
            "promisor himself, or by the conduct of any other person."
        ),
        "section_header": "Section 124 - Contract of Indemnity defined",
        "page_number": 68,
        "document_title": "Indian Contract Act, 1872",
        "category": "Contract Law",
    },
    # BNS chunk
    {
        "chunk_id": "test_bns_1",
        "text": (
            "Section 318 of the Bharatiya Nyaya Sanhita, 2023 defines cheating as dishonestly or "
            "fraudulently inducing a person to deliver property or to consent to retention of property "
            "by any person. Punishment for cheating includes imprisonment up to 3 years."
        ),
        "section_header": "Section 318 - Cheating",
        "page_number": 112,
        "document_title": "Bharatiya Nyaya Sanhita, 2023",
        "category": "Criminal Law",
    },
]


@pytest.fixture(scope="module")
def test_vector_service(tmp_path_factory):
    """Create a fresh in-memory-like vector service with controlled test data."""
    tmp = str(tmp_path_factory.mktemp("test_chromadb"))
    svc = ChromaVectorService(persist_dir=tmp, collection_name="test_accuracy_collection")

    # Index all test chunks
    for chunk in SAMPLE_CHUNKS:
        doc_meta = {
            "doc_id": f"doc_{chunk['document_title'][:10].replace(' ', '_')}",
            "title": chunk["document_title"],
            "category": chunk["category"],
            "num_pages": 200,
            "uploaded_at": "2026-11-11T00:00:00",
            "summary": "",
            "filename": f"{chunk['document_title']}.pdf",
            "total_chars": len(chunk["text"]),
        }
        chunk_data = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "section_header": chunk["section_header"],
            "page_number": chunk["page_number"],
        }
        svc.add_document_chunks(doc_meta, [chunk_data])

    return svc


@pytest.fixture(scope="module")
def test_rag_service(test_vector_service):
    """RAG service wired to the test vector store (no Gemini API key)."""
    gemini = GeminiLegalService(api_key=None)  # Offline fallback mode
    return RAGLegalService(v_service=test_vector_service, g_service=gemini)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Relevant query returns DPDP results, not unrelated docs
# ─────────────────────────────────────────────────────────────────────────────

def test_dpdp_data_breach_returns_dpdp_results(test_vector_service):
    """Query about DPDP data breach must return DPDP Act chunks, not Contract Act."""
    results = test_vector_service.query_similar(
        "What is the penalty for data breach under the DPDP Act?",
        top_k=3,
    )
    assert len(results) >= 1, "Expected at least 1 result for DPDP data breach query"
    titles = [r["document_title"] for r in results]
    assert any("Digital Personal Data Protection" in t for t in titles), (
        f"Expected DPDP Act results, got: {titles}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Article 21 (Constitution) returns ZERO results (not in KB)
# ─────────────────────────────────────────────────────────────────────────────

def test_article_21_returns_no_results(test_vector_service):
    """Query about Article 21 of the Constitution must return empty (not in KB)."""
    results = test_vector_service.query_similar(
        "What is Article 21 of the Constitution of India?",
        top_k=3,
    )
    assert len(results) == 0, (
        f"Expected 0 results for Article 21 (Constitution not in KB), got {len(results)}: "
        + str([r.get("document_title") + " | " + r.get("section_header") for r in results])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Section mismatch — query for Section 420 IPC returns no results
# ─────────────────────────────────────────────────────────────────────────────

def test_section_420_ipc_returns_no_results(test_vector_service):
    """Section 420 IPC is not indexed; must return empty results."""
    results = test_vector_service.query_similar(
        "What is Section 420 of the Indian Penal Code?",
        top_k=3,
    )
    assert len(results) == 0, (
        f"Expected 0 results for IPC Section 420 (not in KB), got {len(results)}: "
        + str([r.get("document_title") + " | " + r.get("section_header") for r in results])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Section 73 Indian Contract Act returns the correct chunk
# ─────────────────────────────────────────────────────────────────────────────

def test_section_73_contract_act_returns_correct_doc(test_vector_service):
    """Query for Section 73 must return the Contract Act chunk, not DPDP or BNS."""
    results = test_vector_service.query_similar(
        "What does Section 73 of the Indian Contract Act say about breach compensation?",
        top_k=3,
    )
    assert len(results) >= 1, "Expected at least 1 result for Section 73 query"
    top_result = results[0]
    assert "Indian Contract Act" in top_result["document_title"], (
        f"Top result should be from Indian Contract Act, got: {top_result['document_title']}"
    )
    assert "73" in top_result["section_header"], (
        f"Top result section header should contain '73', got: {top_result['section_header']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Specific section query does NOT return a different section from same Act
# ─────────────────────────────────────────────────────────────────────────────

def test_section_73_does_not_return_section_124(test_vector_service):
    """Section 73 query must NOT surface Section 124 (different provision, same Act)."""
    results = test_vector_service.query_similar(
        "What is Section 73 of the Indian Contract Act?",
        top_k=5,
    )
    section_headers = [r.get("section_header", "") for r in results]
    for header in section_headers:
        assert "124" not in header, (
            f"Section 124 (indemnity) should not appear in results for a Section 73 query. "
            f"Got headers: {section_headers}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Full RAG pipeline returns "source unavailable" for Constitution query
# ─────────────────────────────────────────────────────────────────────────────

def test_rag_returns_source_unavailable_for_constitution_query(test_rag_service):
    """Full pipeline: Article 21 query must produce source-unavailable message, citations=[]."""
    request = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=3,
        include_disclaimer=False,
    )
    response = test_rag_service.query(request)

    assert response.citations == [], (
        f"Expected empty citations for Article 21, got: {response.citations}"
    )
    assert response.confidence_score == 0.0, (
        f"Expected confidence 0.0 for unavailable source, got: {response.confidence_score}"
    )
    unavailable_keywords = ["not available", "does not contain", "knowledge base", "source not available"]
    answer_lower = response.answer.lower()
    assert any(kw in answer_lower for kw in unavailable_keywords), (
        f"Expected 'source unavailable' message in answer, got: {response.answer[:300]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Full RAG pipeline returns citations for DPDP query
# ─────────────────────────────────────────────────────────────────────────────

def test_rag_returns_citations_for_dpdp_query(test_rag_service):
    """Full pipeline: DPDP data breach query must return at least 1 citation."""
    request = QueryRequest(
        question="What are the penalties for data breach under DPDP Act 2023?",
        top_k=3,
        include_disclaimer=False,
    )
    response = test_rag_service.query(request)

    assert len(response.citations) >= 1, (
        f"Expected at least 1 citation for DPDP breach query, got: {response.citations}"
    )
    assert response.confidence_score > 0.0, (
        f"Expected confidence > 0.0 for DPDP query, got: {response.confidence_score}"
    )
    citation_titles = [c.document_title for c in response.citations]
    assert any("Digital Personal Data Protection" in t for t in citation_titles), (
        f"Expected DPDP Act in citations, got: {citation_titles}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Relevance scores are within valid range [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_relevance_scores_are_valid(test_vector_service):
    """All returned relevance scores must be in [0, 1] range."""
    results = test_vector_service.query_similar(
        "What is a data breach penalty under DPDP?",
        top_k=5,
    )
    for r in results:
        score = r.get("relevance_score", -1)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score} for chunk {r.get('chunk_id')}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Results are sorted by descending relevance score
# ─────────────────────────────────────────────────────────────────────────────

def test_results_are_sorted_by_relevance(test_vector_service):
    """Results must be in descending order of relevance_score."""
    results = test_vector_service.query_similar(
        "DPDP Act obligations Data Fiduciary",
        top_k=5,
    )
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i]["relevance_score"] >= results[i + 1]["relevance_score"], (
                f"Results not sorted: index {i} score {results[i]['relevance_score']} "
                f"< index {i+1} score {results[i+1]['relevance_score']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: BNS cheating section is retrieved for cheating query
# ─────────────────────────────────────────────────────────────────────────────

def test_bns_cheating_section_returned_for_cheating_query(test_vector_service):
    """Query about cheating/fraud under BNS must return BNS chunk."""
    results = test_vector_service.query_similar(
        "What is the punishment for cheating under Bharatiya Nyaya Sanhita?",
        top_k=3,
    )
    assert len(results) >= 1, "Expected BNS cheating result"
    titles = [r["document_title"] for r in results]
    assert any("Bharatiya Nyaya Sanhita" in t for t in titles), (
        f"Expected BNS result for cheating query, got: {titles}"
    )
