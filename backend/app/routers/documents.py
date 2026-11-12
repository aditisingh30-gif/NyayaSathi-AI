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
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
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
                uploaded = supabase_storage.upload_file(
                    file_path=saved_path,
                    storage_path=saved_filename,
                    content_type="application/pdf",
                )
                if not uploaded:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Failed to store document in Supabase Storage.",
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
    """
    doc = vector_service.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = doc.get("filename", "document.pdf")
    storage_filename = f"{doc_id}_{filename}"
    pdf_bytes = None

    # 1. Try Supabase Storage if configured
    if supabase_storage.is_configured:
        try:
            pdf_bytes = supabase_storage.download_file(storage_filename)
        except Exception as e:
            # Fall back to local disk if Supabase download has an issue
            pass

    # 2. Fall back to local uploads directory or sample_data
    if not pdf_bytes:
        local_path = UPLOADS_DIR / storage_filename
        if not local_path.exists():
            # Check if it was one of the sample benchmarks
            from app.config import SAMPLE_DIR
            sample_path = SAMPLE_DIR / filename
            if sample_path.exists():
                local_path = sample_path

        if local_path.exists():
            with open(local_path, "rb") as f:
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
    storage_filename = f"{doc_id}_{filename}" if filename else None

    # Delete from Supabase Storage if configured
    if storage_filename and supabase_storage.is_configured:
        supabase_storage.delete_file(storage_filename)

    # Delete from local uploads if present
    if storage_filename:
        local_path = UPLOADS_DIR / storage_filename
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
