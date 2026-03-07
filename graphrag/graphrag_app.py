"""
graphrag_app.py — Streamlit chat interface for the GraphRAG property recommendation system.

Run with:
    streamlit run graphrag_app.py

The app connects to:
  - Neo4j (shared with kg/ pipeline) for graph queries
  - ChromaDB (shared db at kg/db/chroma_kg) for vector search
  - Gemini for intent parsing and answer generation
"""

import streamlit as st
from loguru import logger

from graphrag_intent import parse_intent
from graphrag_retriever import DualRetriever
from graphrag_answer import generate_answer

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RERA Property Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main { background-color: #0f1117; }

    /* Chat message styling */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* Intent badge */
    .intent-badge {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #a0aec0;
        margin-bottom: 6px;
    }
    .intent-badge strong { color: #63b3ed; }

    /* Project card */
    .project-card {
        background: #1a1d27;
        border: 1px solid #2d3250;
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .project-card h4 { color: #90cdf4; margin: 0 0 6px 0; }

    /* Source tags */
    .tag-both   { background:#276749; color:#9ae6b4; padding:2px 8px; border-radius:12px; font-size:0.72rem; }
    .tag-graph  { background:#2c5282; color:#90cdf4; padding:2px 8px; border-radius:12px; font-size:0.72rem; }
    .tag-vector { background:#553c9a; color:#d6bcfa; padding:2px 8px; border-radius:12px; font-size:0.72rem; }

    /* Sidebar */
    .sidebar-info { color: #718096; font-size: 0.8rem; line-height: 1.6; }

    /* Stat box */
    .stat-box {
        background: #1a1d27;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stat-box .stat-num { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
    .stat-box .stat-lbl { font-size: 0.72rem; color: #718096; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/home.png", width=60)
    st.title("🏠 Property Assistant")
    st.markdown("---")

    st.markdown("**About**")
    st.markdown(
        '<div class="sidebar-info">'
        "AI-powered property recommender using Knowledge Graph + Vector Search (GraphRAG). "
        "Data sourced from real estate project brochures.<br><br>"
        "<b>Powered by:</b> Neo4j · ChromaDB · Gemini"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Example Queries**")
    example_queries = [
        "Show me all available projects",
        "2 BHK in Vinzol with swimming pool",
        "Affordable 1 BHK near school",
        "Luxury villa in West Ahmedabad",
        "3 BHK apartment with clubhouse and gym",
        "Flats near hospital in Ahmedabad",
    ]
    for eq in example_queries:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state["preset_query"] = eq

    st.markdown("---")
    show_debug = st.toggle("Show Debug Info", value=False)
    show_context = st.toggle("Show Retrieved Context", value=False)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    with st.spinner("Connecting to knowledge graph…"):
        try:
            st.session_state.retriever = DualRetriever()
            st.session_state.retriever_ok = True
        except Exception as e:
            st.session_state.retriever_ok = False
            st.session_state.retriever_error = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏘️ RERA Property Recommendation System")
st.markdown(
    "Ask me anything about residential projects — location, BHK, amenities, nearby landmarks, or just describe what you want."
)

if not st.session_state.get("retriever_ok"):
    st.error(
        f"⚠️ Could not connect to the database: {st.session_state.get('retriever_error', 'Unknown error')}. "
        "Make sure Neo4j is running and ChromaDB exists at the configured path."
    )
    st.stop()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show metadata if it was stored
        if msg["role"] == "assistant" and show_debug and "debug" in msg:
            with st.expander("🔍 Intent & Debug"):
                st.json(msg["debug"])
        if msg["role"] == "assistant" and show_context and "context" in msg:
            with st.expander("📄 Retrieved Context"):
                st.text(msg["context"])

# ── Input ──────────────────────────────────────────────────────────────────────
# Handle preset query from sidebar buttons
preset = st.session_state.pop("preset_query", None)
user_input = st.chat_input("Describe the property you're looking for…")
query = preset or user_input

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.status("Searching properties…", expanded=True) as status:
            # Step 1: Parse intent
            status.write("🧠 Understanding your query…")
            intent = parse_intent(query)

            # Step 2: Retrieve
            status.write("🔍 Searching knowledge graph and vector index…")
            retriever: DualRetriever = st.session_state.retriever
            context_text, results = retriever.retrieve_and_assemble(intent, query)

            # Step 3: Generate answer
            status.write("✍️ Generating recommendations…")
            answer = generate_answer(query, context_text, results)

            status.update(
                label=f"✅ Found {len(results)} project(s)", state="complete", expanded=False
            )

        # ── Display answer ─────────────────────────────────────────────────────
        st.markdown(answer)

        # ── Show retrieved project cards ───────────────────────────────────────
        if results:
            with st.expander(f"📋 {len(results)} Retrieved Project(s)", expanded=False):
                cols = st.columns(2)
                for i, proj in enumerate(results):
                    tag_cls = {"both": "tag-both", "graph": "tag-graph", "vector": "tag-vector"}.get(
                        proj.source, "tag-graph"
                    )
                    tag_label = {"both": "Graph + Vector", "graph": "Graph Match", "vector": "Vector Match"}.get(
                        proj.source, proj.source
                    )
                    unit_types = list({u.get("unit_type", "?") for u in proj.units}) if proj.units else []
                    amenity_str = ", ".join(proj.amenities[:4]) or "—"

                    with cols[i % 2]:
                        st.markdown(
                            f'<div class="project-card">'
                            f'<h4>{proj.project_name} <span class="{tag_cls}">{tag_label}</span></h4>'
                            f'📍 {proj.neighbourhood}, {proj.city}<br>'
                            f'🏗️ {proj.developer}<br>'
                            f'🏠 {", ".join(unit_types) if unit_types else "—"}<br>'
                            f'✨ {amenity_str}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # ── Debug / context expanders ──────────────────────────────────────────
        if show_debug:
            with st.expander("🔍 Parsed Intent"):
                st.json(intent.model_dump())

        if show_context:
            with st.expander("📄 Retrieved Context"):
                st.text(context_text)

        # Store in session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "debug": intent.model_dump(),
            "context": context_text,
        })
