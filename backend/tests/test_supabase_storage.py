import io
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings, UPLOADS_DIR
from app.services.supabase_storage import SupabaseStorageService, supabase_storage
from app.services.vector_service import vector_service
from app.services.pdf_service import PDFProcessingService
from app.services.recovery_service import recover_corpus_from_storage


@pytest.fixture
def test_client():
    return TestClient(app)


class TestSupabaseStorageIntegration:
    """Test suite for Supabase storage upload, download/retrieval, error handling, and recovery."""

    def test_supabase_storage_not_configured_graceful(self):
        """When Supabase is not configured, service methods should fail gracefully or return None."""
        unconfigured = SupabaseStorageService(supabase_url="", secret_key="", bucket_name="")
        assert not unconfigured.is_configured
        success, err = unconfigured.upload_bytes(b"dummy content", "test.pdf")
        assert success is False
        assert err == "Supabase storage is not configured"
        assert unconfigured.download_file("test.pdf") is None
        assert unconfigured.delete_file("test.pdf") is False
        assert unconfigured.list_files() == []

    def test_supabase_upload_and_download_flow(self):
        """Test successful upload and download using a mocked Supabase storage service with sb_secret_* key."""
        service = SupabaseStorageService(
            supabase_url="https://mockproject.supabase.co",
            secret_key="sb_secret_sample_key_12345",
            bucket_name="nyayasathi-documents",
        )
        assert service.is_configured

        test_bytes = b"%PDF-1.4 Mock Legal Content for Supabase Storage Test"

        # Mock successful upload response (HTTP 200)
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"Key": "test.pdf"})
            success, err = service.upload_bytes(test_bytes, "test.pdf")
            assert success is True
            assert err is None
            mock_post.assert_called_once()
            # Verify headers: for sb_secret_* keys, key is sent via apikey and NOT Authorization Bearer
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["headers"]["apikey"] == "sb_secret_sample_key_12345"
            assert "Authorization" not in call_kwargs["headers"]

        # Mock successful download response (HTTP 200)
        with patch("httpx.Client.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, content=test_bytes)
            downloaded = service.download_file("test.pdf")
            assert downloaded == test_bytes
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["headers"]["apikey"] == "sb_secret_sample_key_12345"
            assert "Authorization" not in call_kwargs["headers"]

    def test_supabase_url_cleaning_and_normalization(self):
        """Verify SUPABASE_URL is normalized to origin only without any trailing paths like /rest/v1 or /storage/v1."""
        test_cases = [
            ("https://xyzproject.supabase.co/rest/v1", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co/rest/v1/", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co/storage/v1", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co/storage/v1/", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co/api/v1", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co/", "https://xyzproject.supabase.co"),
            ("https://xyzproject.supabase.co", "https://xyzproject.supabase.co"),
            ("  https://xyzproject.supabase.co/rest/v1/  ", "https://xyzproject.supabase.co"),
        ]
        for input_url, expected in test_cases:
            svc = SupabaseStorageService(supabase_url=input_url, secret_key="k", bucket_name="b")
            assert svc.supabase_url == expected, f"Failed for input: {input_url}"

    def test_exact_constructed_storage_urls(self):
        """Verify the exact constructed endpoints for upload, download, delete, and list."""
        svc = SupabaseStorageService(
            supabase_url="https://xyzproject.supabase.co/rest/v1/",
            secret_key="sb_secret_exact_urls",
            bucket_name="nyayasathi-documents",
        )
        assert svc.supabase_url == "https://xyzproject.supabase.co"

        # 1. Object URL (used by upload and download)
        obj_url = svc._object_url("doc_123_sample file.pdf")
        assert obj_url == "https://xyzproject.supabase.co/storage/v1/object/nyayasathi-documents/doc_123_sample%20file.pdf"

        # 2. Delete URL (batch object delete endpoint)
        with patch("httpx.Client.request") as mock_req:
            mock_req.return_value = MagicMock(status_code=200)
            svc.delete_file("doc_123_sample file.pdf")
            call_url = mock_req.call_args[0][1]
            assert call_url == "https://xyzproject.supabase.co/storage/v1/object/nyayasathi-documents"

        # 3. List URL (POST list endpoint)
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: [])
            svc.list_files(paginate=False)
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://xyzproject.supabase.co/storage/v1/object/list/nyayasathi-documents"

    def test_bucket_name_with_spaces_and_special_chars_is_encoded(self):
        """Verify bucket name is properly URL-encoded in all endpoints."""
        svc = SupabaseStorageService(
            supabase_url="https://xyzproject.supabase.co",
            secret_key="sb_secret_key",
            bucket_name="my legal documents bucket",
        )
        assert svc.encoded_bucket == "my%20legal%20documents%20bucket"
        assert svc._object_url("test.pdf") == "https://xyzproject.supabase.co/storage/v1/object/my%20legal%20documents%20bucket/test.pdf"

    def test_no_secret_keys_in_logs_on_http_failure(self, caplog):
        """Verify that when Supabase returns an error like 404 PGRST125, secret keys are never logged."""
        import logging
        caplog.set_level(logging.DEBUG)

        secret_canary = "sb_secret_super_secret_canary_key_9999"
        svc = SupabaseStorageService(
            supabase_url="https://xyzproject.supabase.co",
            secret_key=secret_canary,
            bucket_name="nyayasathi-documents",
        )

        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=404,
                text='{"code": "PGRST125", "message": "Invalid path specified in request URL"}'
            )
            svc.list_files()

        # Check logs captured
        all_logs = " ".join([rec.message for rec in caplog.records])
        assert secret_canary not in all_logs, "Secret key was leaked into application logs!"
        assert "PGRST125" in all_logs or "404" in all_logs
        """Verify legacy JWT tokens still receive both apikey and Authorization Bearer headers."""
        service = SupabaseStorageService(
            supabase_url="https://mockproject.supabase.co",
            secret_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_payload",
            bucket_name="nyayasathi-documents",
        )
        headers = service._get_headers()
        assert headers["apikey"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_payload"
        assert headers["Authorization"] == "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_payload"

    def test_list_files_pagination_support(self):
        """Verify list_files paginates through multiple pages when bucket has >100 files."""
        service = SupabaseStorageService(
            supabase_url="https://mockproject.supabase.co",
            secret_key="sb_secret_paginated_key",
            bucket_name="nyayasathi-documents",
        )
        # Create 250 mock files across 3 pages: 100, 100, 50
        page1 = [{"name": f"doc_{i:03d}_file.pdf"} for i in range(100)]
        page2 = [{"name": f"doc_{i:03d}_file.pdf"} for i in range(100, 200)]
        page3 = [{"name": f"doc_{i:03d}_file.pdf"} for i in range(200, 250)]

        responses = [
            MagicMock(status_code=200, json=lambda: page1),
            MagicMock(status_code=200, json=lambda: page2),
            MagicMock(status_code=200, json=lambda: page3),
        ]

        with patch("httpx.Client.post", side_effect=responses) as mock_post:
            files = service.list_files(paginate=True)
            assert len(files) == 250
            assert mock_post.call_count == 3
            # Check offsets passed in requests
            calls = mock_post.call_args_list
            assert calls[0][1]["json"]["offset"] == 0
            assert calls[1][1]["json"]["offset"] == 100
            assert calls[2][1]["json"]["offset"] == 200

    def test_delete_file_targets_exact_object(self):
        """Verify delete_file targets the exact object path and doesn't delete similarly named files."""
        service = SupabaseStorageService(
            supabase_url="https://mockproject.supabase.co",
            secret_key="sb_secret_test",
            bucket_name="nyayasathi-documents",
        )
        exact_target = "doc_123_contract.pdf"

        with patch("httpx.Client.request") as mock_req:
            mock_req.return_value = MagicMock(status_code=200)
            res = service.delete_file(exact_target)
            assert res is True
            mock_req.assert_called_once()
            call_kwargs = mock_req.call_args[1]
            # Ensure prefixes body contains only the exact string
            assert call_kwargs["json"]["prefixes"] == [exact_target]

        # Verify empty/whitespace storage path is rejected without making a request
        with patch("httpx.Client.request") as mock_req2:
            assert service.delete_file("") is False
            assert service.delete_file("   ") is False
            mock_req2.assert_not_called()

    def test_supabase_upload_failure_handling(self, test_client):
        """When Supabase upload fails with 502/network error, API handles it cleanly."""
        pdf_bytes = io.BytesIO(b"%PDF-1.4 sample content")
        files = {"file": ("test_statute.pdf", pdf_bytes, "application/pdf")}

        with patch.object(supabase_storage, "is_configured", True):
            with patch.object(supabase_storage, "upload_file", return_value=(False, "HTTP 404 - {\"error\": \"Bucket not found\"}")):
                response = test_client.post(
                    "/api/documents/upload",
                    files=files,
                    data={"title": "Test Statute", "category": "Statutory Act"}
                )
                assert response.status_code == 502
                assert "Failed to store document in Supabase Storage" in response.json()["detail"]
                assert "Bucket not found" in response.json()["detail"]

    def test_supabase_upload_auto_provisions_missing_bucket(self):
        """When upload receives 404 Bucket not found, ensure_bucket() provisions it and upload retries."""
        service = SupabaseStorageService(
            supabase_url="https://mockproject.supabase.co",
            secret_key="sb_secret_auto_prov",
            bucket_name="nyayasathi-documents",
        )
        assert service.is_configured

        test_bytes = b"%PDF-1.4 sample content"

        # 1st call: POST /storage/v1/object/... -> 404 Bucket not found
        # 2nd call: GET /storage/v1/bucket/... (check in ensure_bucket) -> 404
        # 3rd call: POST /storage/v1/bucket (create bucket) -> 201
        # 4th call: POST /storage/v1/object/... (retry upload) -> 200
        mock_upload_fail = MagicMock(status_code=404, text='{"statusCode": "404", "error": "Bucket not found"}')
        mock_bucket_check = MagicMock(status_code=404, text='{"error": "Bucket not found"}')
        mock_bucket_create = MagicMock(status_code=201, json=lambda: {"name": "nyayasathi-documents"})
        mock_upload_retry = MagicMock(status_code=200, json=lambda: {"Key": "test.pdf"})

        with patch("httpx.Client.post", side_effect=[mock_upload_fail, mock_bucket_create, mock_upload_retry]) as mock_post:
            with patch("httpx.Client.get", return_value=mock_bucket_check) as mock_get:
                success, err = service.upload_bytes(test_bytes, "test.pdf")
                assert success is True
                assert err is None
                assert mock_post.call_count == 3
                mock_get.assert_called_once()

    def test_document_download_endpoint_from_supabase(self, test_client, tmp_path):
        """Verify the /api/documents/{doc_id}/download endpoint retrieves file from Supabase first."""
        # Create a mock document in vector service registry
        test_doc_id = "doc_test_download_99"
        mock_pdf_content = b"%PDF-1.4 genuine downloaded statutory content from supabase"
        vector_service._doc_registry[test_doc_id] = {
            "doc_id": test_doc_id,
            "filename": "Court_Judgment_2026.pdf",
            "title": "Supreme Court Landmark Judgment",
            "category": "Court Judgment",
            "num_pages": 1,
            "num_chunks": 1,
            "file_size_bytes": len(mock_pdf_content),
            "uploaded_at": "2026-11-12T16:00:00Z",
            "summary": "Mock summary",
        }
        vector_service._save_registry()

        try:
            with patch.object(supabase_storage, "is_configured", True):
                with patch.object(supabase_storage, "download_file", return_value=mock_pdf_content) as mock_dl:
                    resp = test_client.get(f"/api/documents/{test_doc_id}/download")
                    assert resp.status_code == 200
                    assert resp.content == mock_pdf_content
                    assert resp.headers["content-type"] == "application/pdf"
                    assert "Court_Judgment_2026.pdf" in resp.headers["content-disposition"]
                    mock_dl.assert_called_once_with(f"{test_doc_id}_Court_Judgment_2026.pdf")
        finally:
            if test_doc_id in vector_service._doc_registry:
                del vector_service._doc_registry[test_doc_id]
                vector_service._save_registry()

    def test_recovery_service_rebuilds_chromadb_from_supabase(self, tmp_path):
        """Verify that when ChromaDB is wiped, recovery_service pulls PDFs from Supabase and re-indexes them."""
        # Generate a valid PDF in memory
        sample_pdf_path = tmp_path / "doc_testrec99_Recovery_Act.pdf"
        PDFProcessingService.generate_legal_pdf(
            title="Recovery Test Act, 2026",
            sections=[("Section 1. Recovery", "This Act provides automatic recovery procedures from object storage.")],
            output_path=str(sample_pdf_path)
        )
        with open(sample_pdf_path, "rb") as f:
            valid_pdf_bytes = f.read()

        mock_supabase_files = [
            {"name": "doc_testrec99_Recovery_Act.pdf", "created_at": "2026-11-12T16:00:00Z"}
        ]

        # Ensure vector service is cleared
        vector_service.clear()
        assert vector_service.get_stats()["total_documents"] == 0

        with patch.object(supabase_storage, "is_configured", True):
            with patch.object(supabase_storage, "list_files", return_value=mock_supabase_files):
                with patch.object(supabase_storage, "download_file", return_value=valid_pdf_bytes):
                    result = recover_corpus_from_storage()

                    assert result["recovered_documents"] >= 1
                    doc = vector_service.get_document_by_id("doc_testrec99")
                    assert doc is not None
                    assert "Recovery_Act.pdf" in doc["filename"]

                    # Check that the retrieved chunks can be queried via vector similarity
                    results = vector_service.query_similar("recovery procedures from object storage", top_k=3)
                    assert len(results) > 0
                    assert results[0]["document_id"] == "doc_testrec99"

        # Cleanup created file and reload benchmark corpus
        if sample_pdf_path.exists():
            sample_pdf_path.unlink()
        local_cached = UPLOADS_DIR / "doc_testrec99_Recovery_Act.pdf"
        if local_cached.exists():
            local_cached.unlink()

        # Re-populate sample benchmarks so other tests remain clean
        from app.services.sample_data_loader import preload_sample_legal_corpus
        preload_sample_legal_corpus(force=True)
