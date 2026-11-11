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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NyayaSathi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure sample benchmark legal corpus is indexed if store is empty
    logger.info("Initializing NyayaSathi AI Legal & Policy Assistant...")
    stats = vector_service.get_stats()
    if stats["total_documents"] == 0:
        logger.info("ChromaDB is empty. Auto-indexing benchmark legal statutes and contracts...")
        count = preload_sample_legal_corpus(force=False)
        logger.info(f"Auto-indexed {count} benchmark legal documents.")
    else:
        logger.info(f"Loaded existing legal corpus ({stats['total_documents']} documents, {stats['total_chunks']} chunks).")
    yield
    logger.info("NyayaSathi backend shutting down gracefully.")


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
