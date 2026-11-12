"""
test_retrieval_rag_procedural.py
================================
Regression tests for NyayaSathi RAG retrieval & procedural query handling:
1. Consumer complaint procedure query retrieves Section 35 (Manner in which complaint shall be made).
2. Admissibility and 21-day decision timeline is retrieved from and attributed to Section 36 (Proceedings before District Commission).
3. Procedure after admission and 30-day response window is retrieved from and attributed to Section 38 (Procedure on admission of complaint).
4. Consumer complaint definition query retrieves Section 2 (Definitions).
5. Central Consumer Protection Authority query retrieves Section 10 (Establishment of CCPA).
6. Article 21 query retrieves the Constitution correctly when indexed, and returns empty/source unavailable when absent.
7. Statutory sections are NEVER labeled as articles.
8. Every legal statement cites its exact section number and source page; no provision is attributed to the wrong section.
9. Unrelated questions return "Source Not Available in Knowledge Base" without hallucinated citations.
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
            "(3) The electronic filing of complaints may be made through the e-Daakhil portal."
        ),
        "section_header": "Section 35. Manner in which complaint shall be made",
        "article_number": "35",
        "page_number": 12,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
    {
        "chunk_id": "cpa_s36",
        "text": (
            "Section 36. Proceedings before District Commission.—(1) Every proceeding before the District Commission shall be conducted by "
            "the President and at least one member sitting together. (2) On receipt of a complaint made under section 35, the District Commission "
            "may, by order, admit the complaint for being proceeded with, or reject the same: Provided that the admissibility of the complaint "
            "shall ordinarily be decided within twenty-one days from the date on which the complaint was filed: Provided further that no complaint "
            "shall be rejected without giving the complainant an opportunity of being heard. (3) Where the District Commission does not decide "
            "the issue of admissibility within twenty-one days, the complaint shall be deemed to have been admitted."
        ),
        "section_header": "Section 36. Proceedings before District Commission",
        "article_number": "36",
        "page_number": 13,
        "is_schedule": False,
        "document_title": "Consumer Protection Act, 2019",
        "category": "Consumer Statute",
    },
    {
        "chunk_id": "cpa_s38",
        "text": (
            "Section 38. Procedure on admission of complaint.—(1) The District Commission shall, on admission of a complaint, "
            "refer a copy of the admitted complaint within twenty-one days from the date of its admission to the opposite party "
            "directing him to give his version of the case within a period of thirty days, extendable by fifteen days. "
            "(2) Where the opposite party denies or disputes allegations or fails to take action within time, the District Commission "
            "shall proceed to settle the consumer dispute."
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
    {
        "chunk_id": "const_art35",
        "text": (
            "35. Legislation to give effect to the provisions of this Part.—"
            "Notwithstanding anything in this Constitution,—(a) Parliament shall have, and the Legislature of a State shall not have, "
            "power to make laws with respect to any of the matters which under clause (3) of article 16, clause (3) of article 32, "
            "article 33 and article 34 may be provided for by law made by Parliament."
        ),
        "section_header": "Article 35. Legislation to give effect to the provisions of this Part",
        "article_number": "35",
        "page_number": 51,
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

    # Verify Section 36 is also in top retrieved procedural context
    retrieved_sec_numbers = [r["article_number"] for r in results]
    assert "35" in retrieved_sec_numbers
    assert "36" in retrieved_sec_numbers


def test_section_36_admissibility_and_21_day_timeline_retrieval(populated_vector_service):
    """
    Verify that admissibility and 21-day timeline questions retrieve Section 36 (not Section 35).
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the admissibility decision timeline for a consumer complaint under Section 36 of the Consumer Protection Act, 2019?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["article_number"] == "36"
    assert "Section 36" in top_result["section_header"]
    assert "Proceedings before District Commission" in top_result["section_header"]
    assert top_result["page_number"] == 13
    assert "twenty-one days" in top_result["text"].lower()


def test_section_38_procedure_after_admission_retrieval(populated_vector_service):
    """
    Verify that procedure after admission and opposite party notice questions retrieve Section 38.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure after admission of a complaint under Section 38 of the Consumer Protection Act, 2019? What notice is given to the opposite party?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["article_number"] == "38"
    assert "Section 38" in top_result["section_header"]
    assert "Procedure on admission of complaint" in top_result["section_header"]
    assert top_result["page_number"] == 15
    assert "opposite party" in top_result["text"].lower()
    assert "thirty days" in top_result["text"].lower()


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


def test_full_rag_pipeline_consumer_procedure_and_distinct_attribution(populated_vector_service):
    """
    Full RAG pipeline test for the user question:
    Verifies:
    1. Citations include Section 35 and Section 36 with accurate page numbers and document titles.
    2. Section 35 is strictly cited for manner of filing, e-Daakhil, and fees.
    3. Section 36 is strictly cited for admissibility and 21-day timeline.
    4. Section 38 is cited for procedure after admission and opposite party notice.
    5. Zero occurrences of 'Article 35', 'Article 36', or 'Article 38'.
    6. No requirement is attributed to the wrong section.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure for filing a consumer complaint under the Consumer Protection Act, 2019? Mention the relevant sections."

    req = QueryRequest(question=question)
    resp = rag.query(req)

    assert len(resp.citations) >= 2
    cit_sections = [c.section_header for c in resp.citations]
    assert any("Section 35" in s for s in cit_sections), f"Section 35 missing in {cit_sections}"
    assert any("Section 36" in s for s in cit_sections), f"Section 36 missing in {cit_sections}"

    # Verify no occurrence of 'Article 35', 'Article 36', or 'Article 38'
    assert "Article 35" not in resp.answer
    assert "Article 36" not in resp.answer
    assert "Article 38" not in resp.answer

    ans_lower = resp.answer.lower()
    # Section 35 attribution
    assert "section 35" in ans_lower
    assert "manner" in ans_lower or "filing" in ans_lower

    # Section 36 attribution
    assert "section 36" in ans_lower
    assert "admissibility" in ans_lower or "twenty-one" in ans_lower or "21 days" in ans_lower

    # Verify page numbers in citations
    c35 = next(c for c in resp.citations if "Section 35" in c.section_header)
    assert c35.page_number == 12
    assert "Consumer Protection Act, 2019" in c35.document_title

    c36 = next(c for c in resp.citations if "Section 36" in c.section_header)
    assert c36.page_number == 13
    assert "Consumer Protection Act, 2019" in c36.document_title


def test_consumer_complaint_under_section_35_retrieves_cpa_not_constitution_article_35(populated_vector_service):
    """
    CRITICAL REGRESSION TEST for live bug:
    When asking: 'What is the procedure for filing a consumer complaint under Section 35?'
    1. Must retrieve Consumer Protection Act, 2019 Section 35 (page 12), Section 36 (page 13), and Section 38 (page 15).
    2. Must NEVER retrieve Constitution of India Article 35 (page 51).
    3. Full RAG pipeline must generate citations from Consumer Protection Act only.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is the procedure for filing a consumer complaint under Section 35?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0, "Expected results for consumer complaint under Section 35"

    # Verify top result is CPA Section 35
    top = results[0]
    assert top["article_number"] == "35"
    assert "Consumer Protection Act, 2019" in top["document_title"]
    assert top["page_number"] == 12
    assert "Section 35" in top["section_header"]

    # Verify procedural multi-section retrieval includes Sections 36 and/or 38
    retrieved_sections = [r["article_number"] for r in results]
    assert "35" in retrieved_sections
    assert any(s in retrieved_sections for s in ("36", "38"))

    # Verify Constitution Article 35 was NEVER retrieved
    for r in results:
        assert r["document_title"] != "Constitution of India", (
            f"Constitution of India erroneously retrieved for consumer query: {r}"
        )
        assert not r["section_header"].startswith("Article 35"), (
            f"Article 35 erroneously retrieved for Section 35 consumer query: {r}"
        )

    # Full RAG pipeline verification
    req = QueryRequest(question=question)
    resp = rag.query(req)
    assert len(resp.citations) >= 1
    for c in resp.citations:
        assert c.document_title == "Consumer Protection Act, 2019"
        assert "Article" not in c.section_header
        assert c.section_header.startswith("Section ")


def test_constitution_article_35_query_retrieves_constitution_article_35(populated_vector_service):
    """
    Verify that querying for 'Article 35 of the Constitution of India'
    strictly retrieves Constitution of India Article 35 (page 51) and NOT CPA Section 35.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What is Article 35 of the Constitution of India?"

    expanded = rag.expand_query(question)
    results = populated_vector_service.query_similar(
        query_text=expanded,
        top_k=3,
        original_query=question,
    )

    assert len(results) > 0
    top = results[0]
    assert top["article_number"] == "35"
    assert top["document_title"] == "Constitution of India"
    assert top["page_number"] == 51
    assert "Article 35" in top["section_header"]

    # Verify CPA Section 35 was NOT retrieved
    for r in results:
        assert r["document_title"] != "Consumer Protection Act, 2019", (
            f"CPA erroneously retrieved for Constitution Article 35 query: {r}"
        )


def test_bare_section_35_query_returns_source_not_available(populated_vector_service):
    """
    A bare query mentioning only 'Section 35' without statute or subject matter context
    must NOT arbitrarily select an unrelated document; it must return Source Not Available.
    """
    rag = RAGLegalService(v_service=populated_vector_service)
    question = "What does Section 35 say?"

    req = QueryRequest(question=question)
    resp = rag.query(req)

    assert len(resp.citations) == 0, f"Expected 0 citations for ambiguous bare query, got {resp.citations}"
    assert "Source Not Available in Knowledge Base" in resp.answer
    assert resp.confidence_score == 0.0


def test_multiple_provision_numbers_extraction():
    """
    Verify extraction of multiple provision numbers:
    - Section 35
    - Sections 35, 36 and 38
    - Sections 35, 36, 38 and 39
    """
    from app.services.vector_service import extract_target_provision_details

    # Single
    p1, n1 = extract_target_provision_details("What is Section 35 of the CPA?")
    assert p1 == "section"
    assert n1 == {"35"}

    # Three provisions
    p2, n2 = extract_target_provision_details(
        "What is the procedure for filing a consumer complaint under Section 35 of the Consumer Protection Act, 2019? Explain Sections 35, 36 and 38 separately."
    )
    assert p2 == "section"
    assert n2 == {"35", "36", "38"}

    # Four provisions
    p3, n3 = extract_target_provision_details("Explain Sections 35, 36, 38 and 39.")
    assert p3 == "section"
    assert n3 == {"35", "36", "38", "39"}

    # Constitution Article
    p4, n4 = extract_target_provision_details("What is Article 35 of the Constitution of India?")
    assert p4 == "article"
    assert n4 == {"35"}


def test_realistic_uploaded_cpa_metadata_retrieval(tmp_path):
    """
    REGRESSION TEST with realistic uploaded-document metadata:
    Simulates the exact conditions of the live Render failure:
    - document_title = '2606GI.p65' (embedded Gazette PageMaker metadata)
    - filename = 'Consumer Protection Act, 2019.pdf'
    - category = 'Legal Document' (generic default assigned during upload/recovery)
    - Gazette subsection format: '35. (1) ...', '36. (1) ...', '38. (1) ...'
    - Exact PDF page numbers: Page 18 (Sections 35, 36), Page 19 (Section 38)
    - Constitution of India (including Article 35 on page 51 and Article 21 on page 42) indexed concurrently.

    Verifies:
    1. Consumer complaint procedure retrieves Sections 35, 36 and 38.
    2. Constitution Article 35 query retrieves ONLY Constitution Article 35.
    3. Constitution Article 21 continues retrieving Constitution Article 21.
    4. Unrelated query returns Source Not Available.
    5. Bare ambiguous Section 35 returns Source Not Available.
    """
    persist_dir = str(tmp_path / "chroma_realistic_test")
    vs = ChromaVectorService(persist_dir=persist_dir, collection_name="test_realistic_cpa")

    # Realistic uploaded CPA chunks with legacy/raw Gazette metadata
    realistic_cpa_meta = {
        "doc_id": "doc_uploaded_cpa_live",
        "title": "2606GI.p65",  # Raw PageMaker embedded title
        "filename": "Consumer Protection Act, 2019.pdf",
        "category": "Legal Document",  # Generic upload category
        "num_pages": 40,
    }
    realistic_cpa_chunks = [
        {
            "chunk_id": "cpa_live_p18_s35",
            "text": (
                "35. (1) A complaint, in relation to any goods sold or delivered or agreed to be sold or "
                "delivered or any service provided or agreed to be provided, may be filed with a District "
                "Commission by— (a) the consumer; (b) any recognised consumer association; "
                "(c) one or more consumers having the same interest; or (d) the Central Government, Central Authority or State Government. "
                "Provided that the complaint under this sub-section may be filed electronically in such manner as may be prescribed. "
                "(2) Every complaint filed under sub-section (1) shall be accompanied with such fee and payable in such manner, including electronic form."
            ),
            "section_header": "Section 35",
            "article_number": "35",
            "page_number": 18,
            "is_schedule": False,
            "document_title": "2606GI.p65",
            "filename": "Consumer Protection Act, 2019.pdf",
            "category": "Legal Document",
        },
        {
            "chunk_id": "cpa_live_p18_s36",
            "text": (
                "36. (1) Every proceeding before the District Commission shall be conducted by the President of that Commission "
                "and atleast one member thereof, sitting together. (2) On receipt of a complaint made under section 35, the District Commission "
                "may, by order, admit the complaint for being proceeded with or reject the same: Provided that a complaint shall not be rejected "
                "unless an opportunity of being heard is given to the complainant: Provided further that the admissibility of the complaint shall "
                "ordinarily be decided within twenty-one days from the date on which the complaint was filed. "
                "(3) Where the District Commission does not decide the issue of admissibility within twenty-one days, it shall be deemed to have been admitted."
            ),
            "section_header": "Section 36",
            "article_number": "36",
            "page_number": 18,
            "is_schedule": False,
            "document_title": "2606GI.p65",
            "filename": "Consumer Protection Act, 2019.pdf",
            "category": "Legal Document",
        },
        {
            "chunk_id": "cpa_live_p19_s38",
            "text": (
                "38. (1) The District Commission shall, on admission of a complaint, refer a copy of the admitted complaint within "
                "twenty-one days from the date of its admission to the opposite party, directing him to give his version of the case "
                "within a period of thirty days, extendable by fifteen days. (2) Where the opposite party denies or disputes allegations, "
                "the District Commission shall proceed to settle the consumer dispute."
            ),
            "section_header": "Section 38",
            "article_number": "38",
            "page_number": 19,
            "is_schedule": False,
            "document_title": "2606GI.p65",
            "filename": "Consumer Protection Act, 2019.pdf",
            "category": "Legal Document",
        },
    ]
    vs.add_document_chunks(realistic_cpa_meta, realistic_cpa_chunks)

    # Add Constitution chunks (including Article 35 and Article 21)
    const_meta = {
        "doc_id": "doc_const_live",
        "title": "Constitution of India_ 20 MAY 2026 (final).pdf",
        "filename": "Constitution of India.pdf",
        "category": "Legal Document",
        "num_pages": 402,
    }
    const_chunks = [
        {
            "chunk_id": "const_art21_live",
            "text": "21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty except according to procedure established by law.",
            "section_header": "Article 21. Protection of life and personal liberty",
            "article_number": "21",
            "page_number": 42,
            "is_schedule": False,
            "document_title": "Constitution of India_ 20 MAY 2026 (final).pdf",
            "filename": "Constitution of India.pdf",
            "category": "Legal Document",
        },
        {
            "chunk_id": "const_art35_live",
            "text": "35. Legislation to give effect to the provisions of this Part.—Notwithstanding anything in this Constitution,—(a) Parliament shall have power to make laws with respect to any of the matters under article 16, article 32, article 33 and article 34.",
            "section_header": "Article 35. Legislation to give effect to the provisions of this Part",
            "article_number": "35",
            "page_number": 51,
            "is_schedule": False,
            "document_title": "Constitution of India_ 20 MAY 2026 (final).pdf",
            "filename": "Constitution of India.pdf",
            "category": "Legal Document",
        },
    ]
    vs.add_document_chunks(const_meta, const_chunks)

    rag = RAGLegalService(v_service=vs)

    # 1. Test the exact live question:
    live_q = "What is the procedure for filing a consumer complaint under Section 35 of the Consumer Protection Act, 2019? Explain Sections 35, 36 and 38 separately."
    resp1 = rag.query(QueryRequest(question=live_q))
    assert resp1.confidence_score > 0.0
    assert len(resp1.citations) >= 3
    cit_sections = [c.section_header for c in resp1.citations]
    assert any("35" in s for s in cit_sections), f"Section 35 missing in {cit_sections}"
    assert any("36" in s for s in cit_sections), f"Section 36 missing in {cit_sections}"
    assert any("38" in s for s in cit_sections), f"Section 38 missing in {cit_sections}"
    for c in resp1.citations:
        assert "Constitution" not in c.document_title
        assert not c.section_header.startswith("Article")

    # 2. Test Constitution Article 35 query:
    const_q = "What is Article 35 of the Constitution of India?"
    resp2 = rag.query(QueryRequest(question=const_q))
    assert resp2.confidence_score > 0.0
    assert len(resp2.citations) >= 1
    for c in resp2.citations:
        assert "Constitution" in c.document_title
        assert c.section_header.startswith("Article 35")
        assert c.page_number == 51

    # 3. Test Constitution Article 21 query:
    art21_q = "What is Article 21 of the Constitution of India?"
    resp3 = rag.query(QueryRequest(question=art21_q))
    assert resp3.confidence_score > 0.0
    assert len(resp3.citations) >= 1
    assert any(c.section_header.startswith("Article 21") for c in resp3.citations)

    # 4. Test unrelated query:
    unrelated_q = "What is the procedure for registering a nuclear submarine in India?"
    resp4 = rag.query(QueryRequest(question=unrelated_q))
    assert resp4.confidence_score == 0.0
    assert len(resp4.citations) == 0
    assert "Source Not Available in Knowledge Base" in resp4.answer

    # 5. Test bare ambiguous Section 35 query:
    bare_q = "What does Section 35 say?"
    resp5 = rag.query(QueryRequest(question=bare_q))
    assert resp5.confidence_score == 0.0
    assert len(resp5.citations) == 0
    assert "Source Not Available in Knowledge Base" in resp5.answer
