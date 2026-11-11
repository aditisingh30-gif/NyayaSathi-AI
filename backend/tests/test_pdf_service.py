import os
import pytest
from pathlib import Path
from app.services.pdf_service import PDFProcessingService

def test_pdf_creation_and_extraction(tmp_path):
    pdf_path = str(tmp_path / "test_statute.pdf")
    sections = [
        ("Section 1. Short Title and Commencement", "This Act may be called the Data Governance Act, 2026. It extends to the whole of India."),
        ("Section 2. Definitions", "In this Act, 'Personal Data' means any data about an individual who is identifiable by or in relation to such data."),
        ("Section 3. Obligations of Data Fiduciary", "A Data Fiduciary shall make reasonable security safeguards to prevent personal data breach."),
        ("Clause 4.1. Termination for Convenience", "Either party may terminate this agreement by providing thirty (30) days prior written notice to the other party.")
    ]

    # 1. Generate PDF
    created_path = PDFProcessingService.generate_legal_pdf(
        title="Data Governance Benchmark Act",
        sections=sections,
        output_path=pdf_path
    )
    assert os.path.exists(created_path)
    assert os.path.getsize(created_path) > 500

    # 2. Extract Document
    service = PDFProcessingService(chunk_size=300, chunk_overlap=50)
    metadata, pages = service.extract_document(created_path, filename="test_statute.pdf", doc_id="doc_test_1")

    assert metadata["doc_id"] == "doc_test_1"
    assert metadata["num_pages"] >= 1
    assert "Data Governance" in metadata["title"]
    assert len(pages) >= 1
    assert "Section 1" in pages[0]["text"]

    # 3. Chunk Document
    chunks = service.chunk_document(metadata, pages)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["doc_id"] == "doc_test_1"
        assert c["page_number"] >= 1
        assert len(c["text"]) > 0
        assert "document_title" in c["metadata"]

    print(f"\nPDF Service test passed with {len(chunks)} chunks created successfully!")
