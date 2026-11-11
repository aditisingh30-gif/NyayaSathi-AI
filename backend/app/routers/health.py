import time
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
from app.services.vector_service import vector_service
from app.services.gemini_service import gemini_service
from app.models.schemas import SystemStats

router = APIRouter(prefix="/api", tags=["System & Health"])

SERVER_START_TIME = time.time()


class APIKeyUpdateRequest(BaseModel):
    api_key: str
    model_name: str = "gemini-1.5-flash"


@router.get("/health", response_model=SystemStats)
def get_system_health():
    """Returns real-time status of ChromaDB vector store, Gemini LLM, and document corpus."""
    stats = vector_service.get_stats()
    return SystemStats(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        chroma_collection=stats.get("collection_name", ""),
        gemini_configured=gemini_service.is_configured(),
        active_model=gemini_service.model_name if gemini_service.is_configured() else "nyayasathi-legal-engine (local)",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
    )


@router.post("/settings/gemini-key")
def update_gemini_key(request: APIKeyUpdateRequest):
    """Dynamically configure Gemini API key from the frontend settings UI."""
    if not request.api_key or len(request.api_key.strip()) < 5:
        return {"status": "error", "message": "Invalid API key provided."}

    gemini_service.set_api_key(request.api_key.strip())
    gemini_service.model_name = request.model_name
    return {
        "status": "success",
        "message": f"Gemini API key configured for model {request.model_name}.",
        "gemini_configured": True
    }
