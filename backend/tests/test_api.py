import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.pdf_service import PDFProcessingService


@pytest.fixture(scope="module")
def client():
    c = TestClient(app)
    yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "NyayaSathi" in data["app"]
    assert data["status"] == "online"
    assert "disclaimer" in data


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "active_model" in data


def test_preload_benchmark_legal_corpus(client):
    response = client.post("/api/documents/preload-samples")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify documents list
    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    docs_data = docs_resp.json()
    assert docs_data["total_documents"] >= 4
    titles = [d["title"] for d in docs_data["documents"]]
    assert any("Data Protection" in t for t in titles)
    assert any("Contract" in t for t in titles)


def test_rag_query_endpoint(client):
    query_payload = {
        "question": "What is the penalty for failure to prevent personal data breach under the DPDP Act?",
        "top_k": 3,
        "include_disclaimer": True
    }
    response = client.post("/api/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 20
    assert len(data["citations"]) > 0
    assert data["confidence_score"] > 0
    assert "Disclaimer:" in data["disclaimer"]
    # Check citation contents
    top_cit = data["citations"][0]
    assert "document_title" in top_cit
    assert "page_number" in top_cit


def test_chat_endpoint(client):
    chat_payload = {
        "messages": [
            {"role": "user", "content": "What agreements are declared void under the Indian Contract Act?"}
        ],
        "top_k": 2
    }
    response = client.post("/api/query/chat", json=chat_payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 20
    assert len(data["citations"]) > 0


def test_clause_risk_analysis(client):
    clause_payload = {
        "clause_text": "The Company may terminate this Agreement immediately at its sole discretion without any prior notice and without cause. Vendor shall indemnify Company for all losses, direct or indirect, without limit.",
        "contract_type": "Vendor Master Services Agreement"
    }
    response = client.post("/api/analysis/clause", json=clause_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_risk_score"] in ["HIGH", "CRITICAL"]
    assert len(data["risk_factors"]) >= 1
    assert "improved_clause_draft" in data
    assert "Disclaimer:" in data["disclaimer"]


def test_policy_compliance_audit(client):
    policy_payload = {
        "policy_text": "We provide clear notice of personal data processing. We implement encryption safeguards. Users may request erasure of their records. Our Grievance Officer can be contacted at grievance@legalai.in.",
        "framework": "DPDP_ACT_2023"
    }
    response = client.post("/api/analysis/policy", json=policy_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["framework"] == "DPDP_ACT_2023"
    assert data["overall_compliance_percentage"] >= 50
    assert len(data["checks"]) >= 3


def test_document_upload_and_delete_lifecycle(client, tmp_path):
    # 1. Generate a test PDF
    pdf_file_path = str(tmp_path / "custom_policy.pdf")
    PDFProcessingService.generate_legal_pdf(
        title="Custom Corporate Whistleblower Policy",
        sections=[
            ("Section 1. Reporting Channels", "Employees may report any suspected financial impropriety or regulatory breach directly to the Audit Committee."),
            ("Section 2. Non-Retaliation Protection", "The Company strictly prohibits any retaliatory action against any employee who in good faith reports a violation.")
        ],
        output_path=pdf_file_path
    )

    with open(pdf_file_path, "rb") as f:
        pdf_bytes = f.read()

    # 2. Upload via multipart form
    files = {"file": ("custom_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"title": "Custom Corporate Whistleblower Policy", "category": "Corporate Policy"}
    upload_resp = client.post("/api/documents/upload", files=files, data=data)
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    doc_id = upload_data["doc_id"]
    assert doc_id.startswith("doc_")
    assert upload_data["num_chunks"] >= 1

    # 3. Verify it is searchable
    search_resp = client.post("/api/query", json={"question": "What protection is provided to whistleblowers against retaliation?", "doc_ids": [doc_id]})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert any(c["document_id"] == doc_id for c in search_data["citations"])

    # 4. Delete document
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # 5. Verify document is gone
    get_resp = client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 404
