import sys
import io
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.services.pdf_service import PDFProcessingService


def run_all_tests():
    print("Initializing TestClient...")
    client = TestClient(app)

    print("\n--- 1. Testing Root Endpoint ---")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert "NyayaSathi" in data["app"]
    print("PASS: Root endpoint verified.")

    print("\n--- 2. Testing Health Endpoint ---")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    health = res.json()
    assert health["status"] == "healthy"
    assert health["total_documents"] >= 4
    print(f"PASS: Health check verified. Docs: {health['total_documents']}, Chunks: {health['total_chunks']}")

    print("\n--- 3. Testing Documents Listing ---")
    res = client.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()
    assert docs["total_documents"] >= 4
    titles = [d["title"] for d in docs["documents"]]
    print(f"PASS: Documents listing verified. Found {len(titles)} documents: {titles[:2]}...")

    print("\n--- 4. Testing RAG Legal Query ---")
    query_payload = {
        "question": "What is the penalty for failure to prevent personal data breach under the DPDP Act?",
        "top_k": 3,
        "include_disclaimer": True
    }
    res = client.post("/api/query", json=query_payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    q_data = res.json()
    assert len(q_data["answer"]) > 20
    assert len(q_data["citations"]) > 0
    assert "Disclaimer:" in q_data["disclaimer"]
    print(f"PASS: RAG Query verified. Got {len(q_data['citations'])} citations with confidence {q_data['confidence_score']}.")

    print("\n--- 5. Testing Conversational Chat Endpoint ---")
    chat_payload = {
        "messages": [
            {"role": "user", "content": "What agreements are declared void under the Indian Contract Act?"}
        ],
        "top_k": 2
    }
    res = client.post("/api/query/chat", json=chat_payload)
    assert res.status_code == 200
    chat_data = res.json()
    assert "reply" in chat_data
    assert len(chat_data["reply"]) > 20
    print("PASS: Conversational Chat endpoint verified.")

    print("\n--- 6. Testing Contract Clause Risk Analysis ---")
    clause_payload = {
        "clause_text": "The Company may terminate this Agreement immediately at its sole discretion without any prior notice and without cause. Vendor shall indemnify Company for all losses, direct or indirect, without limit.",
        "contract_type": "Vendor Master Services Agreement"
    }
    res = client.post("/api/analysis/clause", json=clause_payload)
    assert res.status_code == 200
    clause_data = res.json()
    assert clause_data["overall_risk_score"] in ["HIGH", "CRITICAL"]
    assert len(clause_data["risk_factors"]) >= 1
    assert "improved_clause_draft" in clause_data
    print(f"PASS: Clause Risk Analysis verified. Risk: {clause_data['overall_risk_score']} with {len(clause_data['risk_factors'])} risk factors.")

    print("\n--- 7. Testing Policy Compliance Audit ---")
    policy_payload = {
        "policy_text": "We provide clear notice of personal data processing. We implement encryption safeguards. Users may request erasure of their records. Our Grievance Officer can be contacted at grievance@legalai.in.",
        "framework": "DPDP_ACT_2023"
    }
    res = client.post("/api/analysis/policy", json=policy_payload)
    assert res.status_code == 200
    pol_data = res.json()
    assert pol_data["framework"] == "DPDP_ACT_2023"
    assert pol_data["overall_compliance_percentage"] >= 50
    print(f"PASS: Policy Compliance Audit verified. Compliance: {pol_data['overall_compliance_percentage']}%.")

    print("\n--- 8. Testing Document Ingestion (PyMuPDF -> Vector Store -> Query -> Delete) ---")
    test_pdf_path = backend_dir / "data" / "temp_test_doc.pdf"
    PDFProcessingService.generate_legal_pdf(
        title="Whistleblower Protection Policy",
        sections=[
            ("Section 1. Reporting Channels", "Employees may report any suspected financial impropriety or regulatory breach directly to the Audit Committee."),
            ("Section 2. Non-Retaliation Protection", "The Company strictly prohibits any retaliatory action against any employee who in good faith reports a violation.")
        ],
        output_path=str(test_pdf_path)
    )

    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("temp_whistleblower.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"title": "Whistleblower Protection Policy", "category": "Corporate Policy"}
    upload_res = client.post("/api/documents/upload", files=files, data=data)
    assert upload_res.status_code == 200
    uploaded_doc = upload_res.json()
    doc_id = uploaded_doc["doc_id"]
    assert uploaded_doc["num_chunks"] >= 1
    print(f"PASS: Ingested test PDF. Doc ID: {doc_id}, Chunks: {uploaded_doc['num_chunks']}")

    # Search against new doc
    search_res = client.post("/api/query", json={"question": "What protection is provided to whistleblowers against retaliation?", "doc_ids": [doc_id]})
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert any(c["document_id"] == doc_id for c in search_data["citations"])
    print("PASS: Verified new document is immediately retrievable via RAG.")

    # Delete doc
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    print("PASS: Document lifecycle cleanup verified.")

    # Clean up disk file
    if test_pdf_path.exists():
        test_pdf_path.unlink()

    print("\n=======================================================")
    print("ALL 8 NYAYASATHI BACKEND INTEGRATION TESTS PASSED 100%!")
    print("=======================================================")


if __name__ == "__main__":
    run_all_tests()
