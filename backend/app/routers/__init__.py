from app.routers.documents import router as documents_router
from app.routers.query import router as query_router
from app.routers.analysis import router as analysis_router
from app.routers.health import router as health_router

__all__ = [
    "documents_router",
    "query_router",
    "analysis_router",
    "health_router",
]
