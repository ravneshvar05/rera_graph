"""
graphrag_config.py — Single source of truth for GraphRAG system configuration.

Reads from .env file in the graphrag/ directory (or inherits from parent).
Import `settings` anywhere:  from graphrag_config import settings

For API key pooling:
  from graphrag_config import build_gemini_pool, build_groq_pool
"""

import os
import threading
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

try:
    import streamlit as st
    # Mirror Streamlit secrets to os.environ so pydantic-settings can read them.
    # NOTE: This handles top-level scalar secrets only. Complex/nested secrets
    # and multi-line key pools are read at call-time via build_gemini_pool().
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

    # ── LLM — single-key fields (backward compat) ─────────────────────────────
    # If you have multiple keys, use GEMINI_API_KEYS / GROQ_API_KEYS instead.
    GEMINI_API_KEY:  str = Field("",  description="Single Gemini API key (backward compat)")
    GEMINI_API_KEYS: str = Field("",  description="Newline- or comma-separated Gemini key pool")
    GEMINI_MODEL:    str = Field("gemini-2.5-flash", description="Gemini model for cypher + answers")
    GEMINI_INTENT_MODEL: str = Field("gemini-2.5-flash", description="Gemini model for intent parsing")

    GROQ_API_KEY:   str = Field("",  description="Single Groq API key (backward compat)")
    GROQ_API_KEYS:  str = Field("",  description="Newline- or comma-separated Groq key pool")
    GROQ_MODEL:     str = Field("llama-3.3-70b-versatile", description="Groq model")

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
    ENABLE_RERANKER:   bool = Field(False, description="Enable cross-encoder re-ranking (slower but more precise for ambiguous queries)")

    # ── Relevance judge toggle ──────────────────────────────────────────────
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


# ── Key Pool ──────────────────────────────────────────────────────────────────

class SharedCounter:
    """Thread-safe global counter for round-robin rotation."""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def next_value(self) -> int:
        with self._lock:
            val = self._value
            self._value += 1
            return val

_gemini_counter = SharedCounter()
_groq_counter = SharedCounter()

class KeyPool:
    """
    Thread-safe round-robin API key pool.

    On each call to .next() the pool returns the next key in rotation.
    If a key fails (rate-limit or auth error), the caller simply calls
    .next() again to get the next key automatically.
    
    If instantiated with a SharedCounter, it will progress a global index
    across requests, even if KeyPool is recreated dynamically.

    Usage:
        pool = build_gemini_pool(api_keys)
        for _ in range(len(pool)):
            key = pool.next()
            try:
                result = call_api(key)
                break
            except RateLimitError:
                continue  # try next key
    """

    def __init__(self, keys: list[str], counter: SharedCounter | None = None):
        self._keys = [k.strip() for k in keys if k and k.strip()]
        self._counter = counter
        self._local_index = 0
        self._lock = threading.Lock()
        self._last_slot = 0  # 1-based slot index of last returned key

    def next(self) -> str | None:
        """Return the next key in round-robin order. Returns None if pool is empty."""
        if not self._keys:
            return None
            
        if self._counter:
            index = self._counter.next_value()
        else:
            with self._lock:
                index = self._local_index
                self._local_index += 1
        
        slot = (index % len(self._keys)) + 1  # 1-based
        self._last_slot = slot
        return self._keys[slot - 1]

    @property
    def last_slot(self) -> int:
        """Returns the 1-based slot index of the most recently returned key."""
        return self._last_slot

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"KeyPool({len(self._keys)} keys)"


def _parse_keys(raw: str) -> list[str]:
    """Parse a newline- or comma-separated string of API keys into a clean list."""
    if not raw:
        return []
    return [k.strip() for k in raw.replace(",", "\n").splitlines() if k.strip()]


def build_gemini_pool(api_keys: dict | None = None) -> KeyPool:
    """
    Build a Gemini KeyPool, reading from (in priority order):

    1. Frontend api_keys dict  — GEMINI_API_KEYS (multi-line) or GEMINI_API_KEY (single)
    2. Streamlit secrets       — read at call-time to avoid import-time race condition
    3. .env / environment      — GEMINI_API_KEYS (comma-sep) or GEMINI_API_KEY

    A single key still works — the pool just has one slot.
    """
    keys: list[str] = []

    # 1. Frontend (Streamlit sidebar entries stored in session_state.user_settings)
    if api_keys:
        raw = api_keys.get("GEMINI_API_KEYS", "")
        keys = _parse_keys(raw)
        if not keys:
            single = api_keys.get("GEMINI_API_KEY", "")
            if single and single.strip():
                keys = [single.strip()]

    # 2. Streamlit secrets — read at call-time (NOT import-time), fixes race condition
    if not keys:
        try:
            import streamlit as st
            raw = st.secrets.get("GEMINI_API_KEYS", "")
            keys = _parse_keys(str(raw))
            if not keys:
                single = st.secrets.get("GEMINI_API_KEY", "")
                if single:
                    keys = [str(single).strip()]
        except Exception:
            pass

    # 3. .env / os.environ via settings
    if not keys:
        keys = _parse_keys(settings.GEMINI_API_KEYS)
    if not keys and settings.GEMINI_API_KEY:
        keys = [settings.GEMINI_API_KEY]

    return KeyPool(keys, counter=_gemini_counter)


def build_groq_pool(api_keys: dict | None = None) -> KeyPool:
    """
    Build a Groq KeyPool, reading from (in priority order):

    1. Frontend api_keys dict  — GROQ_API_KEYS (multi-line) or GROQ_API_KEY (single)
    2. Streamlit secrets       — read at call-time
    3. .env / environment      — GROQ_API_KEYS (comma-sep) or GROQ_API_KEY
    """
    keys: list[str] = []

    # 1. Frontend
    if api_keys:
        raw = api_keys.get("GROQ_API_KEYS", "")
        keys = _parse_keys(raw)
        if not keys:
            single = api_keys.get("GROQ_API_KEY", "")
            if single and single.strip():
                keys = [single.strip()]

    # 2. Streamlit secrets
    if not keys:
        try:
            import streamlit as st
            raw = st.secrets.get("GROQ_API_KEYS", "")
            keys = _parse_keys(str(raw))
            if not keys:
                single = st.secrets.get("GROQ_API_KEY", "")
                if single:
                    keys = [str(single).strip()]
        except Exception:
            pass

    # 3. .env / environment
    if not keys:
        keys = _parse_keys(settings.GROQ_API_KEYS)
    if not keys and settings.GROQ_API_KEY:
        keys = [settings.GROQ_API_KEY]

    return KeyPool(keys, counter=_groq_counter)
