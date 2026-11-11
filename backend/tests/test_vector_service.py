import pytest
from app.services.vector_service import ChromaVectorService

def test_chroma_vector_service(tmp_path):
    persist_dir = str(tmp_path / "test_chroma")
    service = ChromaVectorService(persist_dir=persist_dir, collection_name="test_legal_collection")

    doc_meta = {
        "doc_id": "act_dpdp_2023",
        "title": "Digital Personal Data Protection Act, 2023",
        "category": "Statutory Act",
        "filename": "DPDP_Act_2023.pdf",
        "num_pages": 15,
        "total_chars": 25000,
        "uploaded_at": "2026-11-11T12:00:00Z"
    }

    chunks = [
        {
            "chunk_id": "dpdp_c1",
            "page_number": 3,
            "section_header": "Section 4. Grounds for Processing Personal Data",
            "text": "A person may process the personal data of a Data Principal only in accordance with the provisions of this Act and for a lawful purpose for which the Data Principal has given consent."
        },
        {
            "chunk_id": "dpdp_c2",
            "page_number": 5,
            "section_header": "Section 8. General Obligations of Data Fiduciary",
            "text": "A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards to prevent personal data breach."
        },
        {
            "chunk_id": "dpdp_c3",
            "page_number": 9,
            "section_header": "Section 18. Right to Grievance Redressal",
            "text": "A Data Principal shall have the right to have readily available means of grievance redressal provided by a Data Fiduciary in respect of any act or omission."
        }
    ]

    # 1. Add chunks
    indexed_count = service.add_document_chunks(doc_meta, chunks)
    assert indexed_count == 3
    stats = service.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_documents"] == 1

    # 2. Query similar
    results = service.query_similar("What are the obligations regarding security safeguards and data breach?", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert "Section 8" in top_result["section_header"] or "safeguards" in top_result["text"]
    assert top_result["document_id"] == "act_dpdp_2023"
    assert top_result["relevance_score"] > 0.0

    # 3. List documents
    docs = service.list_documents()
    assert len(docs) == 1
    assert docs[0]["title"] == "Digital Personal Data Protection Act, 2023"

    # 4. Delete document
    deleted = service.delete_document("act_dpdp_2023")
    assert deleted is True
    assert service.get_stats()["total_chunks"] == 0
    assert len(service.list_documents()) == 0

    print("\nChroma Vector Service test passed with full indexing, query, and lifecycle!")
