import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.routers import (
    documents_router,
    query_router,
    analysis_router,
    health_router,
)
from app.services.sample_data_loader import preload_sample_legal_corpus
from app.services.vector_service import vector_service
from app.services.recovery_service import recover_corpus_from_storage
from app.services.sync_service import sync_service
from app.services.supabase_storage import supabase_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NyayaSathi")


async def periodic_sync_loop(interval_seconds: int = 60):
    """
    Periodically checks Supabase Storage for newly uploaded PDFs and synchronizes
    them into ChromaDB without requiring a manual click or backend restart.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if supabase_storage.is_configured:
                # Run sync in thread executor so it does not block the asyncio event loop
                res = await asyncio.to_thread(sync_service.sync_documents)
                if res.get("synced_count", 0) > 0:
                    logger.info(
                        f"Periodic sync completed: {res['synced_count']} new document(s) synchronized."
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in background periodic sync loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure legal corpus is recovered/indexed if store is empty
    logger.info("Initializing NyayaSathi AI Legal & Policy Assistant...")
    stats = vector_service.get_stats()
    if stats["total_documents"] == 0:
        logger.info("ChromaDB is empty. Recovering corpus from storage (Supabase / local / benchmarks)...")
        res = recover_corpus_from_storage()
        logger.info(
            f"Recovery complete: {res['recovered_documents']} documents recovered, "
            f"total corpus now {res['total_documents']} documents ({res['total_chunks']} chunks)."
        )
    else:
        logger.info(f"Loaded existing legal corpus ({stats['total_documents']} documents, {stats['total_chunks']} chunks).")

    # Initial sync for any newly uploaded documents in Supabase
    if supabase_storage.is_configured:
        try:
            sync_res = sync_service.sync_documents()
            logger.info(f"Initial Supabase sync: {sync_res['message']}")
        except Exception as e:
            logger.warning(f"Initial Supabase sync notice: {e}")

    # Launch background periodic sync task
    background_sync_task = asyncio.create_task(periodic_sync_loop(interval_seconds=60))

    yield

    # Shutdown
    logger.info("NyayaSathi backend shutting down gracefully...")
    if background_sync_task:
        background_sync_task.cancel()
        try:
            await background_sync_task
        except asyncio.CancelledError:
            pass
    logger.info("Background tasks stopped.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "NyayaSathi (न्यायसाथी) is an AI Legal & Policy Assistant powered by FastAPI, "
        "Google Gemini, ChromaDB, and PyMuPDF. Features RAG document Q&A, statutory citation grounding, "
        "contract clause risk analysis, and regulatory policy compliance auditing."
    ),
    lifespan=lifespan,
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(analysis_router)


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "disclaimer": settings.LEGAL_DISCLAIMER,
    }
