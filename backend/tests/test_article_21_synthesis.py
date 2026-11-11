"""
test_article_21_synthesis.py
=============================
Regression tests verifying full statutory text preservation and evidence-grounded
synthesis for Article 21 and related constitutional provisions in NyayaSathi.

Verifies:
1. Retrieved citation contains the complete statutory text (not truncated to just header).
2. The generated answer includes the full verbatim statutory provision.
3. Unsupported boilerplate/generic legal implications (e.g. 'Mandatory Compliance: The statutory text stipulates explicit operational responsibilities', 'Risk Mitigation: Regular audit...') are eliminated.
4. Explanations directly reference evidence-grounded terms ('personal liberty', 'procedure established by law').
5. Page numbers (Page 42) and source citations are strictly preserved.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.vector_service import ChromaVectorService
from app.services.rag_service import RAGLegalService
from app.services.gemini_service import GeminiLegalService
from app.models.schemas import QueryRequest

SAMPLE_CONST_CHUNK = [
    {
        "chunk_id": "const_p42_c99",
        "text": "21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "section_header": "Article 21. Protection of life and personal liberty",
        "article_number": "21",
        "page_number": 42,
        "is_schedule": False,
    }
]


@pytest.fixture(scope="module")
def synthesis_rag_service(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("synth_chromadb"))
    v_svc = ChromaVectorService(persist_dir=tmp, collection_name="synth_test_collection")
    doc_meta = {
        "doc_id": "doc_const_synth",
        "title": "Constitution of India",
        "category": "Statutory Act",
        "num_pages": 402,
        "filename": "Constitution of India.pdf",
        "total_chars": 50000,
        "uploaded_at": "2026-11-11T00:00:00",
    }
    v_svc.add_document_chunks(doc_meta, SAMPLE_CONST_CHUNK)
    g_svc = GeminiLegalService(api_key=None)
    return RAGLegalService(v_service=v_svc, g_service=g_svc)


def test_article_21_citation_full_text(synthesis_rag_service):
    """Citation must contain the full statutory text, not just the heading."""
    req = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=1,
        include_disclaimer=False,
    )
    res = synthesis_rag_service.query(req)
    assert len(res.citations) == 1
    cit = res.citations[0]
    assert cit.page_number == 42
    assert cit.section_header == "Article 21. Protection of life and personal liberty"
    # Ensure full text is present
    assert "No person shall be deprived of his life or personal liberty except according to procedure established by law" in cit.text_snippet


def test_article_21_answer_verbatim_statutory_text(synthesis_rag_service):
    """Answer must contain the complete verbatim statutory provision."""
    req = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=1,
        include_disclaimer=False,
    )
    res = synthesis_rag_service.query(req)
    ans = res.answer

    # Must contain verbatim text
    assert "No person shall be deprived of his life or personal liberty except according to procedure established by law" in ans
    assert "Page 42" in ans
    assert "Constitution of India" in ans


def test_article_21_no_generic_boilerplate(synthesis_rag_service):
    """Answer must NOT contain unsupported generic corporate boilerplates."""
    req = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=1,
        include_disclaimer=False,
    )
    res = synthesis_rag_service.query(req)
    ans = res.answer

    # Unsupported corporate boilerplates must be absent
    assert "explicit operational responsibilities for entities" not in ans
    assert "Regular audit and verification against the clauses identified above are strongly advised to preclude breach liability" not in ans
    assert "grievance redressal options, or contractual recourse" not in ans


def test_article_21_evidence_grounded_analysis(synthesis_rag_service):
    """Analysis section must directly reflect evidence-grounded statutory terms."""
    req = QueryRequest(
        question="What is Article 21 of the Constitution of India?",
        top_k=1,
        include_disclaimer=False,
    )
    res = synthesis_rag_service.query(req)
    ans = res.answer

    # Must contain evidence-grounded points
    assert "Protection of Life and Liberty" in ans
    assert "Procedural Requirement" in ans
    assert "procedure established by law" in ans
