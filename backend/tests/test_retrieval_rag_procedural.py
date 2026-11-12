"""
test_retrieval_rag_procedural.py
================================
Regression tests for NyayaSathi RAG retrieval & procedural query handling:
1. Consumer complaint procedure query retrieves Section 35 (Manner in which complaint shall be made).
2. Consumer complaint definition query retrieves Section 2 (Definitions).
3. Central Consumer Protection Authority query retrieves Section 10 (Establishment of CCPA).
4. Article 21 query retrieves the Constitution correctly when indexed, and returns empty/source unavailable when absent.
5. Statutory sections are NEVER labeled as articles.
6. Unrelated questions return "Source Not Available in Knowledge Base" without hallucinated citations.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.vector_service import ChromaVectorService, LegalSemanticEmbeddingFunction
from app.services.rag_service import RAGLegalService
from app.services.gemini_service import GeminiLegalService
from app.services.pdf_service import PDFProcessingService
from app.models.schemas import QueryRequest


CONSUMER_ACT_MOCK_CHUNKS = [
    {
        "chunk_id": "cpa_s2",
        "text": (
            "Section 2. Definitions.—In this Act, unless the context otherwise requires,— "
            "(5) 'complainant' means a consumer, voluntary consumer association, Central or State Government, or Central Authority. "
            "(6) 'complaint' means any allegation in writing made by a complainant to obtain relief under this Act for unfair trade practice, defect in goods, or deficiency in service. "
            "(7) 'consumer' means any person who buys any goods or hires any services for consideration, but does not include commercial resale. "
            "(11) 'deficiency' means any fault, imperfection, or shortcoming in quality, nature, and manner of performance."
        ),
        "section_header": "Section 2. Definitions",
        "article_number": "2",
        "page_number": 1,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
    {
        "chunk_id": "cpa_s10",
        "text": (
            "Section 10. Establishment of Central Consumer Protection Authority.—(1) The Central Government shall, "
            "by notification, establish a Central Authority to be known as the Central Consumer Protection Authority (CCPA) "
            "to regulate matters relating to violation of rights of consumers, unfair trade practices and false or misleading advertisements."
        ),
        "section_header": "Section 10. Establishment of Central Consumer Protection Authority",
        "article_number": "10",
        "page_number": 4,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
    {
        "chunk_id": "cpa_s35",
        "text": (
            "Section 35. Manner in which complaint shall be made.—(1) A complaint in relation to any goods sold or services provided "
            "may be filed with a District Commission by: (a) the consumer; (b) any recognised consumer association; "
            "(c) one or more consumers having the same interest; or (d) the Central Government, Central Authority or State Government. "
            "(2) Every complaint shall be accompanied with such fee and in such manner, including electronic filing, as may be prescribed. "
            "(3) The electronic filing of complaints may be made through the e-Daakhil portal. "
            "(4) On receipt of a complaint, the District Commission may admit or reject the same: "
            "Provided that the admissibility shall ordinarily be decided within twenty-one days from the date of filing."
        ),
        "section_header": "Section 35. Manner in which complaint shall be made",
        "article_number": "35",
        "page_number": 12,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
    {
        "chunk_id": "cpa_s38",
        "text": (
            "Section 38. Procedure on admission of complaint.—(1) The District Commission shall, on admission of a complaint, "
            "refer a copy of the admitted complaint within twenty-one days from the date of its admission to the opposite party "
            "directing him to give his version of the case within thirty days, extendable by fifteen days."
        ),
        "section_header": "Section 38. Procedure on admission of complaint",
        "article_number": "38",
        "page_number": 15,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
]

CONSTITUTION_MOCK_CHUNKS = [
    {
        "chunk_id": "const_art21",
        "text": (
            "21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty "
            "except according to procedure established by law."
        ),
        "section_header": "Article 21. Protection of life and personal liberty",
        "article_number": "21",
        "page_number": 42,
        "is_schedule": False,
        "document_title": "Constitution of India",
        "category": "Constitutional Law",
    },
    {
        "chunk_id": "const_art21a",
        "text": (
            "21A. Right to education.—The State shall provide free and compulsory education to all children "
            "of the age of six to fourteen years in such manner as the State may, by law, determine."
        ),
        "section_header": "Article 21A. Right to education",
        "article_number": "21a",
        "page_number": 42,
        "is_schedule": False,
        "document_title": "Constitution of India",
        "category": "Constitutional Law",
    },
]


@pytest.fixture
def populated_vector_service(tmp_path):
    """Temporary ChromaDB populated with Consumer Protection Act and Constitution chunks."""
    persist_dir = str(tmp_path / "chroma_test_procedural")
    vs = ChromaVectorService(persist_dir=persist_dir, collection_name="test_cpa_collection")

    # Add CPA chunks
    cpa_meta = {
        "doc_id": "cpa_2019",
        "title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
        "filename": "Consumer_Protection_Act_2019.pdf",
    }
    vs.add_document_chunks(cpa_meta, CONSUMER_ACT_MOCK_CHUNKS)

    # Add Constitution chunks
    const_meta = {
        "doc_id": "constitution_india",
        "title": "Constitution of India",
        "category": "Constitutional Law",
        "filename": "Constitution_of_India.pdf",
    }
    vs.add_document_chunks(const_meta, CONSTITUTION_MOCK_CHUNKS)

    return vs


def test_consumer_complaint_procedure_retrieves_section_35(populated_vector_service):
    """
    Core regression test:
    When asking about the procedure for filing a consumer complaint under Consumer Protection Act 2019,
    retrieval MUST return Section 35 as the top result (not Section 2 definitions or Section 10 CCPA).
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure for filing a consumer complaint under the Consumer Protection Act, 2019? Mention the relevant sections."

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0, "Expected at least 1 retrieved result for complaint procedure"
    top_result = results[0]

    # Verify top result is Section 35
    assert top_result["article_number"] == "35", f"Expected Section 35 at rank 1, but got {top_result['section_header']}"
    assert "Section 35" in top_result["section_header"]
    assert "Manner in which complaint shall be made" in top_result["section_header"]
    assert top_result["page_number"] == 12

    # Section 2 (definitions) and Section 10 (CCPA) must not rank above Section 35
    if len(results) > 1:
        assert results[0]["article_number"] == "35"


def test_consumer_complaint_definition_retrieves_section_2(populated_vector_service):
    """
    Querying for definitions (e.g. definition of consumer/complaint) MUST return Section 2 at top rank.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the definition of a consumer under the Consumer Protection Act, 2019?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["article_number"] == "2"
    assert "Section 2" in top_result["section_header"]
    assert "Definitions" in top_result["section_header"]
    assert top_result["page_number"] == 1


def test_ccpa_authority_query_retrieves_section_10(populated_vector_service):
    """
    Querying for Central Consumer Protection Authority MUST return Section 10 at top rank.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the Central Consumer Protection Authority under the Consumer Protection Act, 2019?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["article_number"] == "10"
    assert "Section 10" in top_result["section_header"]
    assert "Central Consumer Protection Authority" in top_result["section_header"]
    assert top_result["page_number"] == 4


def test_article_21_retrieves_constitution_correctly(populated_vector_service):
    """
    Query for Article 21 of the Constitution must return Article 21 (page 42), not Article 21A or CPA sections.
    """
    question = "What is Article 21 of the Constitution of India?"
    results = populated_vector_service.query_similar(
        query_text=question,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["article_number"] == "21"
    assert "Article 21" in top_result["section_header"]
    assert top_result["document_title"] == "Constitution of India"
    assert top_result["page_number"] == 42


def test_statutory_sections_never_labeled_as_articles():
    """
    Statutory Acts (e.g. Consumer Protection Act) must format headers as 'Section', NEVER 'Article'.
    """
    pdf_svc = PDFProcessingService()
    raw_text = "35. Manner in which complaint shall be made.—(1) A complaint may be filed with a District Commission..."

    # Passing a statutory document title should ensure prefix is 'Section'
    chunks = pdf_svc._split_by_article_boundaries(raw_text, doc_title="Consumer Protection Act, 2019")
    assert len(chunks) == 1
    num, header, text = chunks[0]
    assert num == "35"
    assert header.startswith("Section 35"), f"Expected 'Section 35...', got '{header}'"
    assert not header.startswith("Article"), f"Statutory section erroneously labeled as Article: {header}"

    # Passing Constitution title should ensure prefix is 'Article'
    const_text = "21. Protection of life and personal liberty.—No person shall be deprived..."
    const_chunks = pdf_svc._split_by_article_boundaries(const_text, doc_title="Constitution of India")
    assert len(const_chunks) == 1
    c_num, c_header, c_text = const_chunks[0]
    assert c_num == "21"
    assert c_header.startswith("Article 21"), f"Expected 'Article 21...', got '{c_header}'"


def test_unrelated_query_returns_source_not_available(populated_vector_service):
    """
    Unrelated queries (e.g. nuclear submarine registration) have 0 substantive overlap
    and must return empty results from vector service and 'Source Not Available' from RAG.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure for registering a nuclear submarine in India?"

    req = QueryRequest(question=question)
    resp = rag.query(req)

    assert len(resp.citations) == 0, f"Expected 0 citations, got {resp.citations}"
    assert "Source Not Available in Knowledge Base" in resp.answer
    assert resp.confidence_score == 0.0


def test_full_rag_pipeline_consumer_procedure(populated_vector_service):
    """
    Full RAG pipeline test for the user question:
    Verifies citations contain Section 35, accurate page number, accurate document title,
    and answer explains the complaint procedure without calling statutory sections articles.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure for filing a consumer complaint under the Consumer Protection Act, 2019? Mention the relevant sections."

    req = QueryRequest(question=question)
    resp = rag.query(req)

    assert len(resp.citations) > 0
    top_citation = resp.citations[0]
    assert "Section 35" in top_citation.section_header
    assert top_citation.page_number == 12
    assert "Consumer Protection Act, 2019" in top_citation.document_title

    # Verify no occurrence of 'Article 35' in answer or citation headers
    assert "Article 35" not in resp.answer
    assert "Article 35" not in top_citation.section_header

    # Verify procedural terms in response
    ans_lower = resp.answer.lower()
    assert "district commission" in ans_lower
    assert "section 35" in ans_lower
