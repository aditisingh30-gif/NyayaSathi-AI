"""
test_stage2_retrieval.py
========================
Tests for Stage 2 RAG retrieval:
1. Exact article-number lookup prioritizing chunks where article_number matches.
2. Article 21 retrieval retrieves Article 21, not Article 21A or 22.
3. Article 21A retrieval retrieves Article 21A, not Article 21.
4. Article 22 retrieval retrieves Article 22.
5. Schedule/header chunks filtered out.
6. Missing articles return 0 results (no false positive citations).
7. Full RAG pipeline returns 'source unavailable' for missing articles.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.vector_service import ChromaVectorService
from app.services.rag_service import RAGLegalService
from app.services.gemini_service import GeminiLegalService
from app.models.schemas import QueryRequest

CONSTITUTION_MOCK_CHUNKS = [
    {
        "chunk_id": "const_p42_c20",
        "text": "20. Protection in respect of conviction for offences.—(1) No person shall be convicted of any offence except for violation of a law in force...",
        "section_header": "Article 20. Protection in respect of conviction for offences",
        "article_number": "20",
        "page_number": 42,
        "is_schedule": False,
    },
    {
        "chunk_id": "const_p42_c21",
        "text": "21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "section_header": "Article 21. Protection of life and personal liberty",
        "article_number": "21",
        "page_number": 42,
        "is_schedule": False,
    },
    {
        "chunk_id": "const_p42_c21a",
        "text": "2[21A. Right to education.—The State shall provide free and compulsory education to all children of the age of six to fourteen years...",
        "section_header": "Article 21A. Right to education",
        "article_number": "21a",
        "page_number": 42,
        "is_schedule": False,
    },
    {
        "chunk_id": "const_p42_c22",
        "text": "22. Protection against arrest and detention in certain cases.—(1) No person who is arrested shall be detained in custody without being informed...",
        "section_header": "Article 22. Protection against arrest and detention in certain cases",
        "article_number": "22",
        "page_number": 42,
        "is_schedule": False,
    },
    {
        "chunk_id": "const_p299_c804",
        "text": "268 THE CONSTITUTION OF INDIA (Second Schedule) 11. In this Part",
        "section_header": "11. In this Part",
        "article_number": "",
        "page_number": 299,
        "is_schedule": True,
    },
    {
        "chunk_id": "const_p313_c846",
        "text": "THE CONSTITUTION OF INDIA (Sixth Schedule) 1. Subs",
        "section_header": "1. Subs",
        "article_number": "",
        "page_number": 313,
        "is_schedule": True,
    },
]


@pytest.fixture(scope="module")
def stage2_vector_service(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("stage2_chromadb"))
    svc = ChromaVectorService(persist_dir=tmp, collection_name="stage2_test_collection")

    doc_meta = {
        "doc_id": "const_doc_id",
        "title": "Constitution of India",
        "category": "Statutory Act",
        "num_pages": 402,
        "filename": "Constitution_of_India.pdf",
        "total_chars": 100000,
        "uploaded_at": "2026-11-11T00:00:00",
    }
    svc.add_document_chunks(doc_meta, CONSTITUTION_MOCK_CHUNKS)
    return svc


@pytest.fixture(scope="module")
def stage2_rag_service(stage2_vector_service):
    gemini = GeminiLegalService(api_key=None)
    return RAGLegalService(v_service=stage2_vector_service, g_service=gemini)


def test_article_21_exact_retrieval(stage2_vector_service):
    """Article 21 query must return Article 21 on page 42, never 21A, 22, or schedules."""
    results = stage2_vector_service.query_similar(
        "What is Article 21 of the Constitution of India?",
        top_k=3,
        original_query="What is Article 21 of the Constitution of India?",
    )
    assert len(results) >= 1, "Expected at least 1 result for Article 21"
    top = results[0]
    assert top["article_number"] == "21", f"Expected article_number 21, got {top['article_number']}"
    assert top["page_number"] == 42, f"Expected page 42, got {top['page_number']}"
    assert "personal liberty" in top["text"].lower()
    assert "free and compulsory education" not in top["text"].lower()
    assert "arrest and detention" not in top["text"].lower()


def test_article_21a_not_confused_with_21(stage2_vector_service):
    """Article 21A query must return Article 21A (Right to education), NOT Article 21."""
    results = stage2_vector_service.query_similar(
        "What is Article 21A of the Constitution of India?",
        top_k=3,
        original_query="What is Article 21A of the Constitution of India?",
    )
    assert len(results) >= 1, "Expected at least 1 result for Article 21A"
    top = results[0]
    assert top["article_number"] == "21a", f"Expected article_number 21a, got {top['article_number']}"
    assert "Right to education" in top["section_header"]
    assert "free and compulsory education" in top["text"].lower()
    # Must NOT return Article 21 as top chunk
    assert top["text"].strip().startswith("2[21A")


def test_article_22_retrieval(stage2_vector_service):
    """Article 22 query must return Article 22."""
    results = stage2_vector_service.query_similar(
        "What is Article 22 of the Constitution of India?",
        top_k=3,
        original_query="What is Article 22 of the Constitution of India?",
    )
    assert len(results) >= 1
    top = results[0]
    assert top["article_number"] == "22"
    assert "arrest and detention" in top["text"].lower()


def test_schedules_filtered_out_for_article_queries(stage2_vector_service):
    """Schedules (pages 299, 313) must not be returned when searching for Article 21."""
    results = stage2_vector_service.query_similar(
        "What is Article 21 of the Constitution of India?",
        top_k=5,
        original_query="What is Article 21 of the Constitution of India?",
    )
    pages = [r["page_number"] for r in results]
    assert 299 not in pages, "Schedule page 299 returned for Article 21!"
    assert 313 not in pages, "Schedule page 313 returned for Article 21!"


def test_missing_article_returns_empty(stage2_vector_service):
    """An article not in the mock KB (e.g. Article 370) must return 0 results."""
    results = stage2_vector_service.query_similar(
        "What is Article 370 of the Constitution of India?",
        top_k=3,
        original_query="What is Article 370 of the Constitution of India?",
    )
    assert len(results) == 0, f"Expected 0 results for missing Article 370, got {len(results)}"


def test_rag_service_missing_article_no_citations(stage2_rag_service):
    """Full RAG pipeline for missing article returns no citations and source unavailable."""
    req = QueryRequest(
        question="What is Article 370 of the Constitution of India?",
        top_k=3,
        include_disclaimer=False,
    )
    resp = stage2_rag_service.query(req)
    assert resp.citations == []
    assert resp.confidence_score == 0.0
    assert "not available" in resp.answer.lower() or "source not available" in resp.answer.lower()


def test_rag_service_article_21_returns_correct_citation(stage2_rag_service):
    """Full RAG pipeline for Article 21 returns citation with page 42 and correct text."""
    req = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=1,
        include_disclaimer=False,
    )
    resp = stage2_rag_service.query(req)
    assert len(resp.citations) == 1
    cit = resp.citations[0]
    assert cit.page_number == 42
    assert "Article 21" in cit.section_header
    assert "personal liberty" in cit.text_snippet.lower()
