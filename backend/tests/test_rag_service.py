import pytest
from app.services.vector_service import ChromaVectorService
from app.services.gemini_service import GeminiLegalService
from app.services.rag_service import RAGLegalService
from app.models.schemas import QueryRequest

def test_rag_service_pipeline(tmp_path):
    persist_dir = str(tmp_path / "test_rag_chroma")
    v_service = ChromaVectorService(persist_dir=persist_dir, collection_name="test_rag_collection")
    g_service = GeminiLegalService()
    rag = RAGLegalService(v_service=v_service, g_service=g_service)

    # 1. Populate test legal chunk
    doc_meta = {
        "doc_id": "bns_test",
        "title": "Bharatiya Nyaya Sanhita (BNS) 2023",
        "category": "Criminal Statute",
        "filename": "BNS_2023.pdf",
        "num_pages": 10,
        "total_chars": 15000,
        "uploaded_at": "2026-11-11T12:00:00Z"
    }

    chunks = [
        {
            "chunk_id": "bns_c1",
            "page_number": 42,
            "section_header": "Section 318. Cheating",
            "text": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person commits the offence of cheating."
        }
    ]
    v_service.add_document_chunks(doc_meta, chunks)

    # 2. Test query expansion
    expanded = rag.expand_query("What constitutes cheating under Indian law?")
    assert "cheating" in expanded.lower()
    assert "Section 420" in expanded or "318" in expanded

    # 3. Test RAG query execution
    req = QueryRequest(question="What are the essential elements of cheating?", top_k=1)
    response = rag.query(req)

    assert response.query == "What are the essential elements of cheating?"
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "bns_test"
    assert response.citations[0].page_number == 42
    assert "Section 318" in response.citations[0].section_header
    assert "Executive Summary" in response.answer or "Cheating" in response.answer
    assert "Disclaimer:" in response.disclaimer
    assert response.confidence_score > 0.0

    print("\nRAG Legal Service test passed with query expansion, citation grounding, and synthesis!")
