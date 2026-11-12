import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.config import settings, UPLOADS_DIR, SAMPLE_DIR
from app.services.pdf_service import pdf_service
from app.services.vector_service import vector_service
from app.services.sample_data_loader import preload_sample_legal_corpus
from app.services.supabase_storage import supabase_storage

logger = logging.getLogger("NyayaSathi.Recovery")


def recover_corpus_from_storage() -> Dict[str, int]:
    """
    Recovers and rebuilds ChromaDB vector embeddings when the database is empty:
    1. First, checks Supabase Storage for stored user-uploaded PDFs and re-indexes them.
    2. Also checks any local PDFs in UPLOADS_DIR that are not yet in the registry.
    3. Preloads the default benchmark statutory corpus if still empty or as base corpus.

    Returns:
        Dict with counts of recovered documents and chunks.
    """
    stats = vector_service.get_stats()
    existing_docs = {d["doc_id"]: d for d in vector_service.list_documents()}
    total_recovered = 0

    # 1. Recover from Supabase Storage if configured
    if supabase_storage.is_configured:
        try:
            logger.info(f"Checking Supabase bucket '{supabase_storage.bucket}' for stored documents...")
            storage_files = supabase_storage.list_files()
            logger.info(f"Found {len(storage_files)} files in Supabase Storage.")

            for file_item in storage_files:
                name = file_item.get("name", "")
                if not name.lower().endswith(".pdf"):
                    continue

                # Parse doc_id and original filename from '{doc_id}_{filename}' format
                match = re.match(r"^(doc_[a-zA-Z0-9]+)_(.+)$", name)
                if match:
                    doc_id, orig_filename = match.groups()
                else:
                    doc_id = f"doc_{name.replace('.pdf', '')[:10]}"
                    orig_filename = name

                if doc_id in existing_docs:
                    continue

                # Download PDF bytes
                logger.info(f"Recovering document {doc_id} ({orig_filename}) from Supabase...")
                pdf_bytes = supabase_storage.download_file(name)
                if not pdf_bytes:
                    logger.warning(f"Failed to download {name} from Supabase.")
                    continue

                # Cache to local UPLOADS_DIR
                local_path = UPLOADS_DIR / name
                with open(local_path, "wb") as f:
                    f.write(pdf_bytes)

                # Extract and index
                meta, pages = pdf_service.extract_document(
                    file_source=str(local_path),
                    filename=orig_filename,
                    doc_id=doc_id,
                )
                meta["title"] = meta.get("title", orig_filename)
                meta["category"] = "Legal Document"
                meta["uploaded_at"] = file_item.get("created_at") or (datetime.utcnow().isoformat() + "Z")
                meta["summary"] = f"Recovered from Supabase Storage: {meta['num_pages']} pages."

                chunks = pdf_service.chunk_document(meta, pages)
                vector_service.add_document_chunks(meta, chunks)
                total_recovered += 1
                existing_docs[doc_id] = meta
                logger.info(f"Successfully recovered and indexed {doc_id} ({len(chunks)} chunks).")

        except Exception as e:
            logger.error(f"Error recovering documents from Supabase Storage: {e}")

    # 2. Check local UPLOADS_DIR for any PDFs not indexed
    try:
        for local_pdf in UPLOADS_DIR.glob("*.pdf"):
            match = re.match(r"^(doc_[a-zA-Z0-9]+)_(.+)$", local_pdf.name)
            if match:
                doc_id, orig_filename = match.groups()
            else:
                doc_id = f"doc_{local_pdf.stem[:10]}"
                orig_filename = local_pdf.name

            if doc_id in existing_docs:
                continue

            try:
                meta, pages = pdf_service.extract_document(
                    file_source=str(local_pdf),
                    filename=orig_filename,
                    doc_id=doc_id,
                )
                meta["title"] = meta.get("title", orig_filename)
                meta["category"] = "Legal Document"
                meta["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
                meta["summary"] = f"Recovered from local uploads: {meta['num_pages']} pages."

                chunks = pdf_service.chunk_document(meta, pages)
                vector_service.add_document_chunks(meta, chunks)
                total_recovered += 1
                existing_docs[doc_id] = meta
            except Exception as e:
                logger.warning(f"Could not re-index local PDF {local_pdf.name}: {e}")
    except Exception as e:
        logger.warning(f"Error checking local uploads: {e}")

    # 3. If corpus is still completely empty, preload benchmark samples
    current_stats = vector_service.get_stats()
    if current_stats["total_documents"] == 0:
        logger.info("Corpus is still empty after storage scan. Preloading standard benchmarks...")
        preload_sample_legal_corpus(force=False)

    final_stats = vector_service.get_stats()
    return {
        "recovered_documents": total_recovered,
        "total_documents": final_stats["total_documents"],
        "total_chunks": final_stats["total_chunks"],
    }
