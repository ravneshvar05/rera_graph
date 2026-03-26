"""
graphrag_config.py — Single source of truth for GraphRAG system configuration.

Reads from .env file in the graphrag/ directory (or inherits from parent).
Import `settings` anywhere:  from graphrag_config import settings
"""

import os
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

try:
    import streamlit as st
    # Mirror Streamlit secrets to os.environ so pydantic-settings can read them
    for k, v in st.secrets.items():
        if isinstance(v, (str, int, float, bool)):
            os.environ[k] = str(v)
except Exception:
    pass

class GraphRAGSettings(BaseSettings):

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI:      str = Field("bolt://localhost:7687", description="Neo4j bolt URI")
    NEO4J_USER:     str = Field("neo4j",                description="Neo4j username")
    NEO4J_PASSWORD: str = Field(...,                    description="Neo4j password")

    # ── LLM ───────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(...,  description="Gemini API key")
    GEMINI_MODEL:   str = Field("gemini-2.5-flash", description="Gemini model for answers")
    GEMINI_INTENT_MODEL: str = Field("gemini-2.5-flash", description="Gemini model for intent parsing")
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
    VECTOR_TOP_K:      int  = Field(15,  description="How many ChromaDB results to retrieve per doc type")
    GRAPH_MAX_RESULTS: int  = Field(20,  description="Max projects from Neo4j Cypher queries")
    FINAL_TOP_N:       int  = Field(20,  description="Max projects to include in final answer")
    RERANK_MODEL:      str  = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model for re-ranking vector results")
    RERANK_TOP_K:      int  = Field(10,   description="Hard cap on projects sent to LLM judge (controls judge token budget)")
    VECTOR_DISTANCE_THRESHOLD: float = Field(1.3,  description="Max L2 distance from ChromaDB; discard results beyond this (lower = stricter)")
    RERANK_SCORE_THRESHOLD:    float = Field(-3.0, description="Min cross-encoder score; discard re-ranked results below this (only used when ENABLE_RERANKER=true)")

    # ── Cross-encoder toggle ──────────────────────────────────────────────────
    # Set ENABLE_RERANKER=false in .env to disable the cross-encoder entirely.
    # When disabled: ChromaDB distance + VECTOR_DISTANCE_THRESHOLD + RERANK_TOP_K
    # cap act as the pre-filter before the LLM relevance judge.
    # When enabled: cross-encoder re-ranks candidates for higher precision
    # at the cost of ~7-10s CPU inference per query.
    # Re-enable any time without code changes — just set ENABLE_RERANKER=true.
    ENABLE_RERANKER:   bool = Field(False, description="Enable cross-encoder re-ranking (slower but more precise for ambiguous queries)")

    # ── Relevance judge toggle ──────────────────────────────────────────────
    # Set SKIP_VECTOR_JUDGE=true to bypass the LLM relevance judge for vector-only
    # results entirely. This saves judge tokens and shows any project that passed
    # the VECTOR_DISTANCE_THRESHOLD (i.e. >=50-60% semantic similarity).
    # Quality gating is handled purely by VECTOR_DISTANCE_THRESHOLD in this mode.
    SKIP_VECTOR_JUDGE: bool = Field(False, description="Bypass LLM relevance judge for vector-only results; rely on distance threshold alone")

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
