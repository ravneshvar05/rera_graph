"""
graphrag_config.py — Single source of truth for GraphRAG system configuration.

Reads from .env file in the graphrag/ directory (or inherits from parent).
Import `settings` anywhere:  from graphrag_config import settings
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class GraphRAGSettings(BaseSettings):

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI:      str = Field("bolt://localhost:7687", description="Neo4j bolt URI")
    NEO4J_USER:     str = Field("neo4j",                description="Neo4j username")
    NEO4J_PASSWORD: str = Field(...,                    description="Neo4j password")

    # ── LLM ───────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(...,  description="Gemini API key")
    GEMINI_MODEL:   str = Field("gemini-2.0-flash", description="Gemini model for answers")
    GEMINI_INTENT_MODEL: str = Field("gemini-2.0-flash", description="Gemini model for intent parsing")
    GROQ_API_KEY:   str = Field("",  description="Groq API key (optional fallback)")
    GROQ_MODEL:     str = Field("llama-3.3-70b-versatile", description="Groq fallback model")

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    CHROMA_PATH: Path = Field(
        Path("../kg/db/chroma_kg"),
        description="Path to the existing ChromaDB directory (created by kg_ingest.py)",
    )
    CHROMA_COLLECTION_NAME: str = Field("rera_kg_units")
    CHROMA_EMBED_MODEL: str = Field(
        "all-MiniLM-L6-v2",
        description="Must match the model used during kg_ingest.py",
    )

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    VECTOR_TOP_K:      int  = Field(15,  description="How many ChromaDB results to retrieve")
    GRAPH_MAX_RESULTS: int  = Field(30,  description="Max projects from Neo4j Cypher queries")
    FINAL_TOP_N:       int  = Field(10,  description="Max projects to include in final answer")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("CHROMA_PATH", mode="before")
    @classmethod
    def _resolve_path(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        return p


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = GraphRAGSettings()  # type: ignore[call-arg]
