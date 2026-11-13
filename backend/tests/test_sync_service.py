import io
import json
import threading
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings, UPLOADS_DIR
from app.services.sync_service import DocumentSyncService, sync_service
from app.services.supabase_storage import supabase_storage
from app.services.vector_service import vector_service


@pytest.fixture
def client():
    return TestClient(app)


class TestDocumentSyncService:
    """Test suite for Supabase document discovery, pagination, concurrency, and indexing."""

    def test_derive_doc_identifiers_app_format(self):
        """Files uploaded via the app have doc_xxxx_filename.pdf format."""
        storage_name = "doc_a1b2c3d4e5_Consumer_Protection_Act_2019.pdf"
        doc_id, orig_filename = DocumentSyncService.derive_doc_identifiers(storage_name)
        assert doc_id == "doc_a1b2c3d4e5"
        assert orig_filename == "Consumer_Protection_Act_2019.pdf"

    def test_derive_doc_identifiers_direct_upload_with_spaces_and_special_chars(self):
        """Files uploaded directly by admins to Supabase dashboard retain their name and get deterministic IDs."""
        raw_name = "Digital Personal Data Protection Act (DPDP) - Final Official 2023.pdf"
        doc_id1, orig_filename1 = DocumentSyncService.derive_doc_identifiers(raw_name)
        assert orig_filename1 == raw_name
        assert doc_id1.startswith("doc_")
        assert len(doc_id1) == 14  # 'doc_' + 10 hex chars

        # Deterministic: same file name produces identical doc_id
        doc_id2, orig_filename2 = DocumentSyncService.derive_doc_identifiers(raw_name)
        assert doc_id1 == doc_id2

    def test_sync_not_configured(self):
        """When Supabase storage is not configured, sync returns graceful not_configured status."""
        test_sync = DocumentSyncService()
        with patch.object(supabase_storage, "is_configured", False):
            res = test_sync.sync_documents()
            assert res["status"] == "not_configured"
            assert res["synced_count"] == 0
            assert "Supabase Storage is not configured" in res["message"]

    def test_sync_concurrency_lock(self):
        """Simultaneous sync calls should be blocked by the concurrency lock."""
        test_sync = DocumentSyncService()

        # Simulate lock already acquired by another thread/process
        test_sync._lock.acquire(blocking=False)
        try:
            with patch.object(supabase_storage, "is_configured", True):
                res = test_sync.sync_documents()
                assert res["status"] == "in_progress"
                assert "already running" in res["message"]
        finally:
            test_sync._lock.release()

    def test_sync_documents_discovery_and_indexing(self, tmp_path):
        """Discovers new PDFs from Supabase, extracts, and indexes into vector registry."""
        test_sync = DocumentSyncService()

        mock_files = [
            {
                "name": "Consumer Protection Act 2019 Official.pdf",
                "id": "file-uuid-1",
                "created_at": "2026-11-12T10:00:00Z",
                "metadata": {"size": 1024},
            }
        ]

        dummy_pdf_bytes = b"%PDF-1.4 Mock PDF content for sync test"
        dummy_meta = {
            "doc_id": "doc_test123",
            "title": "Consumer Protection Act 2019 Official",
            "num_pages": 5,
            "total_chars": 500,
        }
        dummy_pages = [{"page_number": 1, "text": "Section 35. Manner in which complaint shall be made."}]
        dummy_chunks = [{
            "chunk_id": "doc_test123_p1_c0",
            "document_id": "doc_test123",
            "text": "Section 35. Manner in which complaint shall be made.",
            "page_number": 1,
            "section_header": "Section 35",
            "article_number": "35",
        }]

        with patch.object(supabase_storage, "is_configured", True), \
             patch.object(supabase_storage, "ensure_bucket", return_value=True), \
             patch.object(supabase_storage, "list_files", return_value=mock_files) as mock_list, \
             patch.object(supabase_storage, "download_file", return_value=dummy_pdf_bytes), \
             patch("app.services.sync_service.pdf_service.extract_document", return_value=(dummy_meta, dummy_pages)), \
             patch("app.services.sync_service.pdf_service.chunk_document", return_value=dummy_chunks), \
             patch.object(vector_service, "add_document_chunks", return_value=1) as mock_add:

            res = test_sync.sync_documents()
            assert res["status"] == "success"
            assert res["synced_count"] == 1
            assert res["failed_count"] == 0
            mock_list.assert_called_with(paginate=True)
            mock_add.assert_called_once()

    def test_duplicate_prevention(self):
        """Already indexed documents should be skipped without redundant re-indexing."""
        test_sync = DocumentSyncService()

        existing_doc = {
            "doc_id": "doc_existing1",
            "filename": "Arbitration_Act.pdf",
            "title": "Arbitration Act",
            "indexing_status": "indexed",
            "storage_path": "Arbitration_Act.pdf",
        }

        mock_files = [
            {"name": "Arbitration_Act.pdf", "created_at": "2026-11-12T10:00:00Z", "metadata": {"size": 2048}}
        ]

        with patch.object(supabase_storage, "is_configured", True), \
             patch.object(supabase_storage, "ensure_bucket", return_value=True), \
             patch.object(supabase_storage, "list_files", return_value=mock_files), \
             patch.object(vector_service, "list_documents", return_value=[existing_doc]), \
             patch.object(supabase_storage, "download_file") as mock_download:

            res = test_sync.sync_documents(force=False)
            assert res["status"] == "success"
            assert res["synced_count"] == 0
            assert res["already_indexed_count"] == 1
            # Download should not be called for already indexed document
            mock_download.assert_not_called()

    def test_sync_failure_registers_failed_status(self):
        """When extraction or download fails, document is registered with status='failed' for retry."""
        test_sync = DocumentSyncService()

        mock_files = [
            {"name": "Corrupted_File.pdf", "created_at": "2026-11-12T10:00:00Z", "metadata": {"size": 500}}
        ]

        with patch.object(supabase_storage, "is_configured", True), \
             patch.object(supabase_storage, "ensure_bucket", return_value=True), \
             patch.object(supabase_storage, "list_files", return_value=mock_files), \
             patch.object(supabase_storage, "download_file", return_value=None), \
             patch.object(vector_service, "list_documents", return_value=[]), \
             patch.object(vector_service, "register_failed_document") as mock_reg_fail:

            res = test_sync.sync_documents()
            assert res["status"] == "success"
            assert res["synced_count"] == 0
            assert res["failed_count"] == 1
            mock_reg_fail.assert_called_once()
            failed_arg = mock_reg_fail.call_args[0][0]
            assert failed_arg["indexing_status"] == "failed"
            assert "Could not download" in failed_arg["error_message"]

    def test_retry_document_success(self):
        """Retrying a failed document re-downloads, extracts, and updates status to 'indexed'."""
        test_sync = DocumentSyncService()

        failed_doc = {
            "doc_id": "doc_failed_99",
            "filename": "Pending_Bill.pdf",
            "title": "Pending Bill",
            "indexing_status": "failed",
            "storage_path": "Pending_Bill.pdf",
        }

        dummy_bytes = b"%PDF-1.4 Recovered bytes"
        dummy_meta = {"doc_id": "doc_failed_99", "title": "Pending Bill", "num_pages": 2}
        dummy_pages = [{"page_number": 1, "text": "Bill text"}]
        dummy_chunks = [{"chunk_id": "c1", "text": "Bill text"}]

        with patch.object(vector_service, "get_document_by_id", return_value=failed_doc), \
             patch.object(supabase_storage, "is_configured", True), \
             patch.object(supabase_storage, "download_file", return_value=dummy_bytes), \
             patch("app.services.sync_service.pdf_service.extract_document", return_value=(dummy_meta, dummy_pages)), \
             patch("app.services.sync_service.pdf_service.chunk_document", return_value=dummy_chunks), \
             patch.object(vector_service, "delete_document", return_value=True), \
             patch.object(vector_service, "add_document_chunks", return_value=1):

            res = test_sync.retry_document("doc_failed_99")
            assert res["status"] == "success"
            assert res["doc_id"] == "doc_failed_99"


class TestDocumentSyncApiEndpoints:
    """Test suite for HTTP API endpoints handling synchronization and status."""

    def test_api_sync_endpoint(self, client):
        """POST /api/documents/sync triggers synchronization."""
        mock_result = {
            "status": "success",
            "synced_count": 2,
            "failed_count": 0,
            "already_indexed_count": 3,
            "total_supabase_files": 5,
            "message": "Synchronization complete: 2 new document(s) indexed.",
            "errors": [],
        }
        with patch.object(sync_service, "sync_documents", return_value=mock_result):
            resp = client.post("/api/documents/sync")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["synced_count"] == 2
            assert data["total_supabase_files"] == 5

    def test_api_sync_status_endpoint(self, client):
        """GET /api/documents/sync/status returns the current sync status."""
        mock_status = {
            "is_syncing": False,
            "last_sync_at": "2026-11-12T12:00:00Z",
            "last_sync_result": {"status": "success", "synced_count": 1},
        }
        with patch.object(sync_service, "get_sync_status", return_value=mock_status):
            resp = client.get("/api/documents/sync/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_syncing"] is False
            assert data["last_sync_at"] == "2026-11-12T12:00:00Z"

    def test_api_retry_document_endpoint(self, client):
        """POST /api/documents/{doc_id}/retry triggers retry for specific doc."""
        mock_retry = {
            "status": "success",
            "doc_id": "doc_failed_1",
            "title": "Failed Act",
            "num_pages": 4,
            "num_chunks": 8,
            "message": "Successfully re-indexed 'Failed Act'.",
        }
        with patch.object(sync_service, "retry_document", return_value=mock_retry):
            resp = client.post("/api/documents/doc_failed_1/retry")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["num_chunks"] == 8

    def test_api_delete_document_cleans_exact_storage_path(self, client):
        """DELETE /api/documents/{doc_id} deletes from Supabase using exact storage_path."""
        mock_doc = {
            "doc_id": "doc_direct_upload_1",
            "filename": "Direct Upload Document.pdf",
            "title": "Direct Upload Document",
            "storage_path": "Direct Upload Document.pdf",
            "indexing_status": "indexed",
        }
        with patch.object(vector_service, "get_document_by_id", return_value=mock_doc), \
             patch.object(supabase_storage, "is_configured", True), \
             patch.object(supabase_storage, "delete_file", return_value=True) as mock_delete, \
             patch.object(vector_service, "delete_document", return_value=True):

            resp = client.delete("/api/documents/doc_direct_upload_1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            mock_delete.assert_any_call("Direct Upload Document.pdf")
