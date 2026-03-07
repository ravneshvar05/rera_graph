"""
config.py — Single source of truth for all configuration.
Reads from .env file. Import `settings` anywhere in the project.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(..., description="Gemini API key")
    GEMINI_MODEL: str = Field("gemini-2.5-flash", description="Gemini model for query planning & answers")

    GROQ_API_KEY: str = Field("", description="Groq API key (optional fallback)")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", description="Groq model")

    # ── Paths ─────────────────────────────────────────────────────────────────
    JSON_OUTPUT_DIR: Path = Field(
        Path("./output"),
        description="Folder containing brochure JSON files from gemini.py",
    )
    DB_DIR: Path = Field(
        Path("./db"),
        description="Folder where SQLite + ChromaDB are stored",
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_COLLECTION_NAME: str = Field("rera_units_v2")
    CHROMA_EMBED_MODEL: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model for embeddings",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    INGEST_BATCH_SIZE: int = Field(200, description="ChromaDB upsert batch size")

    # ── Search ────────────────────────────────────────────────────────────────
    MAX_SQL_CANDIDATES: int = Field(1000, description="Max rows from SQL before vector rerank")
    MAX_VECTOR_RESULTS: int = Field(50,  description="Max results from ChromaDB")
    DEFAULT_TOP_K: int = Field(5,        description="Default top-K projects to return")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE: str = Field("RERA Project Finder")
    LOG_LEVEL: str = Field("INFO")

    # ── Derived paths (computed, not from .env) ────────────────────────────
    @property
    def sqlite_path(self) -> Path:
        return self.DB_DIR / "projects_v2.db"

    @property
    def chroma_path(self) -> Path:
        return self.DB_DIR / "chroma_v2"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()

# Ensure required directories exist
settings.DB_DIR.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)
