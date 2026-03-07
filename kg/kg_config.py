"""
kg_config.py — Single source of truth for all KG system configuration.

Reads from .env file in the kg/ directory.
Import `settings` anywhere:  from kg_config import settings
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class KGSettings(BaseSettings):

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI:      str = Field("bolt://localhost:7687", description="Neo4j bolt URI")
    NEO4J_USER:     str = Field("neo4j",                description="Neo4j username")
    NEO4J_PASSWORD: str = Field(...,                    description="Neo4j password")

    # ── LLM ───────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(...,  description="Gemini API key")
    GEMINI_MODEL:   str = Field("gemini-2.0-flash", description="Gemini model")
    GROQ_API_KEY:   str = Field("",  description="Groq API key (optional fallback)")
    GROQ_MODEL:     str = Field("llama-3.3-70b-versatile", description="Groq model")

    # ── Paths ──────────────────────────────────────────────────────────────────
    JSON_OUTPUT_DIR: Path = Field(
        Path("../output"),
        description="Folder containing brochure JSON files (relative to kg/ dir)",
    )
    CHROMA_PATH: Path = Field(
        Path("./db/chroma_kg"),
        description="ChromaDB persistence directory",
    )
    CHROMA_COLLECTION_NAME: str = Field("rera_kg_units")
    CHROMA_EMBED_MODEL: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model name for ChromaDB embeddings",
    )

    # ── Ingestion performance ──────────────────────────────────────────────────
    INGEST_WORKERS:    int = Field(8,   description="Thread pool size for parallel ingestion")
    INGEST_BATCH_SIZE: int = Field(200, description="ChromaDB upsert batch size")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("JSON_OUTPUT_DIR", "CHROMA_PATH", mode="before")
    @classmethod
    def _resolve_path(cls, v: str | Path) -> Path:
        """Resolve relative paths relative to this file's directory."""
        p = Path(v)
        if not p.is_absolute():
            # Resolve relative to the kg/ directory where this file lives
            p = (Path(__file__).parent / p).resolve()
        return p


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = KGSettings()  # type: ignore[call-arg]

# ── Ensure directories exist ───────────────────────────────────────────────────
settings.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
Path(Path(__file__).parent / "logs").mkdir(exist_ok=True)
