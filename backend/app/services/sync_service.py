import os
import re
import time
import logging
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.config import settings, UPLOADS_DIR
from app.services.pdf_service import pdf_service
from app.services.vector_service import vector_service
from app.services.supabase_storage import supabase_storage

logger = logging.getLogger("NyayaSathi.Sync")


class DocumentSyncService:
    """
    Manages synchronization between Supabase Storage ('nyayasathi-documents' bucket)
    and NyayaSathi's ChromaDB legal vector corpus:
    - Paginated discovery of all PDFs in Supabase Storage.
    - Detection of newly added or modified PDFs.
    - Thread-safe concurrency lock to prevent duplicate indexing during simultaneous syncs.
    - Graceful error handling and retry mechanism for failed documents.
    - Preserves metadata: filename, document ID, upload date, page count, and source.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.is_syncing: bool = False
        self.last_sync_at: Optional[str] = None
        self.last_sync_result: Optional[Dict[str, Any]] = None

    def get_sync_status(self) -> Dict[str, Any]:
        """Return the current sync state."""
        return {
            "is_syncing": self.is_syncing,
            "last_sync_at": self.last_sync_at,
            "last_sync_result": self.last_sync_result,
        }

    @staticmethod
    def derive_doc_identifiers(storage_name: str) -> Tuple[str, str]:
        """
        Derive clean doc_id and original filename from a storage object name.
        Handles:
        1. App-uploaded format: '{doc_id}_{filename}' -> (doc_id, filename)
        2. Direct Supabase upload: '{filename}' -> (doc_<hash>, filename)
        Preserves original filename, handling spaces and special characters.
        """
        match = re.match(r"^(doc_[a-zA-Z0-9]+)_(.+)$", storage_name)
        if match:
            doc_id, orig_filename = match.groups()
            return doc_id, orig_filename

        # Direct upload to Supabase bucket without doc_ prefix:
        # Create a deterministic, URL-safe alphanumeric doc_id
        safe_hash = hashlib.md5(storage_name.encode("utf-8")).hexdigest()[:10]
        doc_id = f"doc_{safe_hash}"
        return doc_id, storage_name

    def sync_documents(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute synchronization of PDFs from Supabase Storage into ChromaDB.
        Thread-safe: will reject concurrent sync requests.
        """
        if not supabase_storage.is_configured:
            return {
                "status": "not_configured",
                "synced_count": 0,
                "failed_count": 0,
                "already_indexed_count": 0,
                "total_supabase_files": 0,
                "message": "Supabase Storage is not configured. Cloud synchronization unavailable.",
                "errors": [],
            }

        # Acquire lock to prevent duplicate concurrent syncs
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            logger.info("Sync requested but another synchronization is already in progress.")
            return {
                "status": "in_progress",
                "synced_count": 0,
                "failed_count": 0,
                "already_indexed_count": 0,
                "total_supabase_files": 0,
                "message": "Synchronization is already running in background.",
                "errors": [],
            }

        self.is_syncing = True
        synced_count = 0
        failed_count = 0
        already_indexed_count = 0
        errors: List[str] = []

        try:
            # Ensure bucket exists
            supabase_storage.ensure_bucket()
            logger.info(f"Starting Supabase Storage synchronization from bucket '{supabase_storage.bucket}'...")

            # 1. Fetch all files from Supabase Storage with pagination
            storage_files = supabase_storage.list_files(paginate=True)
            pdf_files = [f for f in storage_files if f.get("name", "").lower().endswith(".pdf")]
            logger.info(f"Found {len(pdf_files)} PDF files in Supabase Storage.")

            # 2. Map existing documents in vector_service to prevent duplicates
            existing_docs = {d["doc_id"]: d for d in vector_service.list_documents()}
            indexed_storage_paths = {
                d.get("storage_path") for d in existing_docs.values() if d.get("storage_path")
            }
            indexed_filenames = {
                d.get("filename") for d in existing_docs.values()
                if d.get("filename") and d.get("indexing_status") == "indexed"
            }

            for file_item in pdf_files:
                storage_name = file_item.get("name", "")
                if not storage_name:
                    continue

                doc_id, orig_filename = self.derive_doc_identifiers(storage_name)

                # Check if already indexed
                already_indexed = (
                    (doc_id in existing_docs and existing_docs[doc_id].get("indexing_status") == "indexed")
                    or storage_name in indexed_storage_paths
                    or (orig_filename in indexed_filenames and not force)
                )

                if already_indexed and not force:
                    already_indexed_count += 1
                    continue

                # 3. New or unindexed PDF discovered -> Download & Process
                logger.info(f"Syncing new document: {storage_name} (ID: {doc_id})...")
                try:
                    pdf_bytes = supabase_storage.download_file(storage_name)
                    if not pdf_bytes:
                        raise RuntimeError(f"Could not download file '{storage_name}' from Supabase Storage.")

                    # Cache to local UPLOADS_DIR
                    local_cache_name = f"{doc_id}_{orig_filename}" if not storage_name.startswith(f"{doc_id}_") else storage_name
                    local_path = UPLOADS_DIR / local_cache_name
                    with open(local_path, "wb") as f:
                        f.write(pdf_bytes)

                    # Extract text and structure
                    meta, pages = pdf_service.extract_document(
                        file_source=str(local_path),
                        filename=orig_filename,
                        doc_id=doc_id,
                    )

                    file_size = len(pdf_bytes)
                    meta["doc_id"] = doc_id
                    meta["filename"] = orig_filename
                    meta["category"] = "Legal Document"
                    meta["uploaded_at"] = file_item.get("created_at") or (datetime.utcnow().isoformat() + "Z")
                    meta["file_size_bytes"] = file_size
                    meta["summary"] = f"Synchronized from Supabase Storage: {meta['num_pages']} pages."
                    meta["indexing_status"] = "indexed"
                    meta["error_message"] = None
                    meta["storage_path"] = storage_name
                    meta["source"] = "supabase"

                    # Chunk and add to ChromaDB
                    chunks = pdf_service.chunk_document(meta, pages)
                    if not chunks:
                        raise RuntimeError(f"No text chunks could be extracted from '{orig_filename}'.")

                    vector_service.add_document_chunks(meta, chunks)
                    synced_count += 1
                    indexed_storage_paths.add(storage_name)
                    indexed_filenames.add(orig_filename)
                    existing_docs[doc_id] = meta
                    logger.info(f"Successfully synchronized '{orig_filename}' ({len(chunks)} chunks indexed).")

                except Exception as doc_err:
                    err_msg = str(doc_err)
                    logger.error(f"Failed to synchronize document '{storage_name}': {err_msg}")
                    failed_count += 1
                    errors.append(f"{orig_filename}: {err_msg}")

                    # Register failed status in registry so user can see failure and retry
                    failed_meta = {
                        "doc_id": doc_id,
                        "filename": orig_filename,
                        "title": Path(orig_filename).stem.replace("_", " ").title(),
                        "category": "Legal Document",
                        "num_pages": 0,
                        "num_chunks": 0,
                        "file_size_bytes": file_item.get("metadata", {}).get("size", 0),
                        "uploaded_at": file_item.get("created_at") or (datetime.utcnow().isoformat() + "Z"),
                        "summary": f"Failed to index: {err_msg}",
                        "indexing_status": "failed",
                        "error_message": err_msg,
                        "storage_path": storage_name,
                        "source": "supabase",
                    }
                    vector_service.register_failed_document(failed_meta)

            result = {
                "status": "success",
                "synced_count": synced_count,
                "failed_count": failed_count,
                "already_indexed_count": already_indexed_count,
                "total_supabase_files": len(pdf_files),
                "message": (
                    f"Synchronization complete: {synced_count} new document(s) indexed, "
                    f"{already_indexed_count} already indexed, {failed_count} failed."
                ),
                "errors": errors,
            }
            self.last_sync_at = datetime.utcnow().isoformat() + "Z"
            self.last_sync_result = result
            return result

        except Exception as e:
            logger.error(f"Error during Supabase document synchronization: {e}")
            err_result = {
                "status": "error",
                "synced_count": synced_count,
                "failed_count": failed_count,
                "already_indexed_count": already_indexed_count,
                "total_supabase_files": 0,
                "message": f"Synchronization error: {str(e)}",
                "errors": [str(e)],
            }
            self.last_sync_result = err_result
            return err_result

        finally:
            self.is_syncing = False
            self._lock.release()

    def retry_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retry indexing for a specific document that previously failed.
        """
        doc = vector_service.get_document_by_id(doc_id)
        if not doc:
            raise ValueError(f"Document with ID '{doc_id}' not found.")

        storage_path = doc.get("storage_path")
        orig_filename = doc.get("filename", "document.pdf")
        if not storage_path:
            storage_path = f"{doc_id}_{orig_filename}"

        logger.info(f"Retrying indexing for document {doc_id} ('{orig_filename}')...")

        # 1. Download file bytes from Supabase
        pdf_bytes = None
        if supabase_storage.is_configured:
            try:
                pdf_bytes = supabase_storage.download_file(storage_path)
            except Exception as e:
                logger.warning(f"Could not download '{storage_path}' from Supabase: {e}")

        # Fallback to local cache if Supabase download was empty
        local_path = UPLOADS_DIR / f"{doc_id}_{orig_filename}"
        if not pdf_bytes and local_path.exists():
            with open(local_path, "rb") as f:
                pdf_bytes = f.read()

        if not pdf_bytes:
            err_msg = f"PDF file '{orig_filename}' could not be downloaded from storage."
            vector_service.update_document_status(doc_id, "failed", err_msg)
            return {
                "status": "failed",
                "doc_id": doc_id,
                "message": err_msg,
            }

        # Save to local cache
        with open(local_path, "wb") as f:
            f.write(pdf_bytes)

        try:
            meta, pages = pdf_service.extract_document(
                file_source=str(local_path),
                filename=orig_filename,
                doc_id=doc_id,
            )
            meta["doc_id"] = doc_id
            meta["filename"] = orig_filename
            meta["category"] = doc.get("category", "Legal Document")
            meta["uploaded_at"] = doc.get("uploaded_at") or (datetime.utcnow().isoformat() + "Z")
            meta["file_size_bytes"] = len(pdf_bytes)
            meta["summary"] = f"Indexed {meta['num_pages']} pages."
            meta["indexing_status"] = "indexed"
            meta["error_message"] = None
            meta["storage_path"] = storage_path
            meta["source"] = doc.get("source", "supabase")

            chunks = pdf_service.chunk_document(meta, pages)
            if not chunks:
                raise RuntimeError(f"No text chunks could be extracted from '{orig_filename}'.")

            # Remove any partial old chunks before re-indexing
            vector_service.delete_document(doc_id)
            vector_service.add_document_chunks(meta, chunks)

            logger.info(f"Retry succeeded for {doc_id}: {len(chunks)} chunks indexed.")
            return {
                "status": "success",
                "doc_id": doc_id,
                "title": meta["title"],
                "num_pages": meta["num_pages"],
                "num_chunks": len(chunks),
                "message": f"Successfully re-indexed '{meta['title']}' ({len(chunks)} chunks).",
            }
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Retry failed for document {doc_id}: {err_msg}")
            vector_service.update_document_status(doc_id, "failed", err_msg)
            return {
                "status": "failed",
                "doc_id": doc_id,
                "message": f"Retry failed: {err_msg}",
            }


# Global singleton instance
sync_service = DocumentSyncService()
