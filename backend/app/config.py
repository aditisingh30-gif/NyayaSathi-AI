import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

# Configurable persistent storage directory (Render Persistent Disk / local)
# Defaults to backend/data if STORAGE_DIR is not set
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR = STORAGE_DIR
CHROMA_DIR = DATA_DIR / "chromadb"
UPLOADS_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "app" / "sample_data"

# Ensure runtime directories exist safely
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    APP_NAME: str = "NyayaSathi AI Legal & Policy Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Gemini LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

    # Vector & File Storage Settings
    STORAGE_DIR: str = str(STORAGE_DIR)
    CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)
    CHROMA_COLLECTION_NAME: str = "nyayasathi_legal_corpus"
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.35

    # Supabase Storage Settings
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY", "")
    SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "nyayasathi-documents")

    # Legal Processing Settings
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 150
    MAX_FILE_SIZE_MB: int = 30

    # Legal Disclaimer
    LEGAL_DISCLAIMER: str = (
        "Disclaimer: NyayaSathi is an AI-powered legal and policy research tool designed for "
        "informational, analytical, and educational purposes. It does not provide formal legal advice, "
        "legal representation, or an attorney-client relationship. Please consult a qualified legal practitioner "
        "before taking any legal action."
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
