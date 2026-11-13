import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Response
from typing import Optional
from app.config import settings, UPLOADS_DIR
from app.services.pdf_service import pdf_service
from app.services.vector_service import vector_service
from app.services.sample_data_loader import preload_sample_legal_corpus
from app.services.supabase_storage import supabase_storage
from app.services.sync_service import sync_service
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentSyncResponse,
    SyncStatusResponse,
    DocumentRetryResponse,
)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_legal_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form("Legal Document"),
):
    """
    Upload a legal PDF document, extract text and statutory clauses using PyMuPDF,
    store the PDF in Supabase Storage and locally, and index vector embeddings into ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for legal ingestion.",
        )

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    saved_filename = f"{doc_id}_{file.filename}"
    saved_path = UPLOADS_DIR / saved_filename

    try:
        # Save file to local disk (uploads dir)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to Supabase Storage if configured
        if supabase_storage.is_configured:
            try:
                uploaded, err_detail = supabase_storage.upload_file(
                    file_path=saved_path,
                    storage_path=saved_filename,
                    content_type="application/pdf",
                )
                if not uploaded:
                    msg = f"Failed to store document in Supabase Storage: {err_detail}" if err_detail else "Failed to store document in Supabase Storage."
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=msg,
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Supabase Storage error: {str(e)}",
                )

        # Process with PyMuPDF
        meta, pages = pdf_service.extract_document(
            file_source=str(saved_path),
            filename=file.filename,
            doc_id=doc_id,
        )

        # Enrich metadata
        meta["title"] = title if title and len(title.strip()) > 1 else meta.get("title", file.filename)
        meta["category"] = category or "General Legal"
        meta["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
        meta["summary"] = f"Extracted {meta['num_pages']} pages and {meta['total_chars']} characters."
        meta["indexing_status"] = "indexed"
        meta["error_message"] = None
        meta["storage_path"] = saved_filename if supabase_storage.is_configured else None
        meta["source"] = "supabase" if supabase_storage.is_configured else "local"

        # Chunk and index
        chunks = pdf_service.chunk_document(meta, pages)
        num_indexed = vector_service.add_document_chunks(meta, chunks)

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            title=meta["title"],
            num_pages=meta["num_pages"],
            num_chunks=num_indexed,
            status="success",
            message=f"Successfully ingested and indexed {num_indexed} legal chunks into ChromaDB.",
        )

    except HTTPException:
        if saved_path.exists():
            saved_path.unlink()
        raise
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}",
        )


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all registered legal documents with metadata."""
    docs = vector_service.list_documents()
    formatted = [DocumentMetadata(**d) for d in docs]
    return DocumentListResponse(
        total_documents=len(formatted),
        documents=formatted,
    )


@router.post("/sync", response_model=DocumentSyncResponse)
def sync_documents_from_storage():
    """
    Synchronize all PDF documents from Supabase Storage ('nyayasathi-documents' bucket)
    into the ChromaDB vector database.
    - Discovers newly added PDFs in Supabase.
    - Extracts, chunks, and indexes them into ChromaDB.
    - Thread-safe: prevents duplicate runs during simultaneous requests.
    """
    result = sync_service.sync_documents(force=False)
    return DocumentSyncResponse(**result)


@router.get("/sync/status", response_model=SyncStatusResponse)
def get_sync_status():
    """Get the current status of document synchronization."""
    status_data = sync_service.get_sync_status()
    return SyncStatusResponse(**status_data)


@router.post("/{doc_id}/retry", response_model=DocumentRetryResponse)
def retry_failed_document(doc_id: str):
    """
    Retry indexing for a specific document that previously failed.
    """
    res = sync_service.retry_document(doc_id)
    return DocumentRetryResponse(**res)


@router.get("/{doc_id}", response_model=DocumentMetadata)
def get_document(doc_id: str):
    """Retrieve metadata for a specific document."""
    doc = vector_service.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentMetadata(**doc)


@router.get("/{doc_id}/download")
def download_document(doc_id: str):
    """
    Download/retrieve the PDF binary for a specific document.
    Attempts to fetch from Supabase Storage first; falls back to local storage.
    Handles both direct Supabase uploads and app-generated prefixes.
    """
    doc = vector_service.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = doc.get("filename", "document.pdf")
    storage_path = doc.get("storage_path") or f"{doc_id}_{filename}"
    pdf_bytes = None

    # 1. Try Supabase Storage if configured
    if supabase_storage.is_configured:
        for candidate_path in (storage_path, f"{doc_id}_{filename}", filename):
            if not candidate_path:
                continue
            try:
                pdf_bytes = supabase_storage.download_file(candidate_path)
                if pdf_bytes:
                    break
            except Exception:
                pass

    # 2. Fall back to local uploads directory or sample_data
    if not pdf_bytes:
        for candidate_name in (storage_path, f"{doc_id}_{filename}", filename):
            if not candidate_name:
                continue
            local_path = UPLOADS_DIR / candidate_name
            if local_path.exists():
                with open(local_path, "rb") as f:
                    pdf_bytes = f.read()
                break

    # 3. Check sample_data
    if not pdf_bytes:
        from app.config import SAMPLE_DIR
        sample_path = SAMPLE_DIR / filename
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                pdf_bytes = f.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=404,
            detail=f"PDF file for document {doc_id} could not be retrieved from storage.",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/pdf",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all indexed vectors from ChromaDB and Supabase Storage."""
    doc = vector_service.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document could not be deleted or was not found.")

    filename = doc.get("filename", "")
    storage_path = doc.get("storage_path") or (f"{doc_id}_{filename}" if filename else None)

    # Delete from Supabase Storage if configured
    if supabase_storage.is_configured:
        if storage_path:
            supabase_storage.delete_file(storage_path)
        if filename and filename != storage_path and not storage_path.startswith("doc_"):
            # Also ensure no dangling legacy prefix
            supabase_storage.delete_file(f"{doc_id}_{filename}")

    # Delete from local uploads if present
    for candidate_name in (storage_path, f"{doc_id}_{filename}" if filename else None, filename):
        if candidate_name:
            local_path = UPLOADS_DIR / candidate_name
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass

    success = vector_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document could not be deleted or was not found.")
    return {"status": "success", "message": f"Document {doc_id} deleted."}


@router.post("/preload-samples")
def trigger_preload_samples():
    """Pre-populate ChromaDB with Indian Legal Acts and Contract benchmarks."""
    count = preload_sample_legal_corpus(force=True)
    return {"status": "success", "message": f"Preloaded {count} benchmark legal documents into ChromaDB."}
