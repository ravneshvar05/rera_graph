"""
graphrag_app.py — Streamlit chat interface for the GraphRAG property recommendation system.

Run with:
    streamlit run graphrag_app.py

The app connects to:
  - Neo4j (shared with kg/ pipeline) for graph queries
  - ChromaDB (shared db at kg/db/chroma_kg) for vector search
  - Gemini for intent parsing and answer generation
"""

import time
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
    /* ── GOOGLE FONTS ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ── GLOBAL STREAMLIT OVERRIDES ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main App Background */
    .stApp { background-color: #0f111a !important; }

    /* Smooth Webkit Scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(144, 205, 244, 0.2); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(144, 205, 244, 0.4); }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151824 0%, #0f111a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Sidebar Content Text adjustments */
    [data-testid="stSidebar"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }

    /* ── BUTTONS & EXPANDERS ── */
    .stButton>button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2) !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(49, 130, 206, 0.3) !important;
        background: linear-gradient(135deg, #4299e1, #3182ce) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        background: rgba(30, 34, 45, 0.4) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }

    /* Status Widget / Loading Expander */
    [data-testid="stStatusWidget"] {
        background: rgba(30, 34, 45, 0.4) !important;
        border: 1px solid rgba(144, 205, 244, 0.15) !important;
        border-radius: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stStatusWidget"] summary {
        border-radius: 16px !important;
        color: #bee3f8 !important;
    }
    [data-testid="stStatusWidget"] summary:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }

    /* ── CHAT INTERFACE ── */
    /* Chat Input Pill */
    [data-testid="stChatInput"] {
        background-color: #1e2230 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 36px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15) !important;
        padding: 6px 16px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(99, 179, 237, 0.6) !important;
        box-shadow: 0 8px 30px rgba(99, 179, 237, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f7fafc !important;
    }

    /* Chat Messages Base */
    [data-testid="stChatMessage"] {
        border-radius: 20px !important;
        padding: 18px 22px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* User Chat Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(66, 153, 225, 0.1), rgba(49, 130, 206, 0.05)) !important;
        border-left: 4px solid #63b3ed !important;
        border-right: none !important;
    }

    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(30, 34, 45, 0.6) !important;
        border-left: 4px solid #9f7aea !important;
    }

    /* ── CUSTOM DOMAIN COMPONENTS ── */

    /* Intent badge */
    .intent-badge {
        background: #232736;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-bottom: 6px;
    }
    .intent-badge strong { color: #90cdf4; }

    /* Project card (fluid hover) */
    .project-card {
        background: linear-gradient(135deg, #1e2230, #252a3a);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #63b3ed;
        border-radius: 16px;
        padding: 18px 22px;
        margin: 12px 0;
        font-size: 0.95rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        color: #e2e8f0;
    }

    .project-card:before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.03) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        transition: left 0.6s ease;
    }

    .project-card:hover {
        transform: translateY(-3px) scale(1.005);
        box-shadow: 0 12px 24px rgba(99, 179, 237, 0.1), 0 0 0 1px rgba(99, 179, 237, 0.2);
        border-left: 4px solid #90cdf4;
    }

    .project-card:hover:before {
        left: 200%;
    }

    .project-card h4 { color: #bee3f8; margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 600; }

    /* LLM Explanation */
    .llm-explanation {
        padding: 0 12px 15px 18px;
        border-left: 2px solid rgba(99, 179, 237, 0.3);
        margin-left: 12px;
        margin-bottom: 24px;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Minimalist Directory Row (>10 results) */
    .project-list-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 18px;
        margin: 6px 0;
        background: rgba(30, 34, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        color: #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .project-list-row:hover {
        background: rgba(36, 41, 56, 0.8);
        border-color: rgba(99, 179, 237, 0.3);
        transform: translateX(3px);
    }
    .project-list-row .pl-title { font-weight: 500; color: #bee3f8; flex: 1.5; }
    .project-list-row .pl-loc { flex: 2; color: #a0aec0; font-size: 0.9rem; }
    .project-list-row .pl-dev { flex: 1; text-align: right; color: #718096; font-size: 0.85rem; }

    /* Source tags */
    .tag-both   { background: rgba(72, 187, 120, 0.15); color: #9ae6b4; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(72, 187, 120, 0.2); }
    .tag-graph  { background: rgba(49, 130, 206, 0.15); color: #90cdf4; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(49, 130, 206, 0.2); }
    .tag-vector { background: rgba(159, 122, 234, 0.15); color: #d6bcfa; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; border: 1px solid rgba(159, 122, 234, 0.2); }

    /* Sidebar */
    .sidebar-info { color: #a0aec0; font-size: 0.85rem; line-height: 1.7; }

    /* Stat box */
    .stat-box {
        background: #1e2230;
        border: 1px solid rgba(255, 255, 255, 0.08); /* Softer */
        border-radius: 12px; /* Softer */
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stat-box .stat-num { font-size: 1.7rem; font-weight: 600; color: #90cdf4; }
    .stat-box .stat-lbl { font-size: 0.8rem; color: #a0aec0; }
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
        content = msg["content"]
        if isinstance(content, dict):
            st.markdown(content.get("general_summary", ""))
            
            projects_data = content.get("projects", [])
            is_large_list = len(projects_data) > 10
            
            for p in projects_data:
                if p.get("html_content"):
                    st.markdown(p["html_content"], unsafe_allow_html=True)
                # Reasoning is only rendered if it exists (not empty) and we are not in large list mode
                reasoning_text = str(p.get("reasoning", "")).strip()
                if reasoning_text and not is_large_list:
                    st.markdown(f'<div class="llm-explanation" markdown="1">\n\n{reasoning_text}\n\n</div>', unsafe_allow_html=True)
                    
            if content.get("conclusion"):
                st.markdown(content.get("conclusion"))
                
            if "llm_metadata" in content:
                meta = content["llm_metadata"]
                st.markdown(f"<div style='font-size: 0.8rem; color: #718096; margin-top: 10px; text-align: right;'>🤖 Engine: {meta.get('model_name', 'unknown')} &nbsp;|&nbsp; 🪙 Tokens: {meta.get('total_tokens', 0)} ({meta.get('prompt_tokens', 0)} prompt + {meta.get('completion_tokens', 0)} completion)</div>", unsafe_allow_html=True)
        else:
            st.markdown(content)

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

            # Mandatory Slot Check: City
            if not intent.city:
                status.update(
                    label="Need more information", state="complete", expanded=False
                )
                answer = "I have many options available! To give you the best recommendations, please mention which **city** you are looking in (e.g., Ahmedabad or Surat)."
                results = []
                context_text = ""
            else:
                # Step 2: Retrieve
                status.write("🔍 Searching knowledge graph and vector index…")
                retriever: DualRetriever = st.session_state.retriever
                context_text, results = retriever.retrieve_and_assemble(intent, query)

                # Step 3: Generate answer
                status.write("✍️ Generating recommendations…")
                answer = generate_answer(query, context_text, results, intent)

                final_count = len(answer.get("projects", [])) if isinstance(answer, dict) else len(results)
                status.update(
                    label=f"✅ Found {final_count} project(s)", state="complete", expanded=False
                )

        # ── Display answer ─────────────────────────────────────────────────────
        if isinstance(answer, dict):
            st.markdown(answer.get("general_summary", ""))
            
            projects_data = answer.get("projects", [])
            is_large_list = len(projects_data) > 10
            
            def get_project_card(proj, tag_c, tag_l):
                u_types = list({u.get("unit_type", "?") for u in proj.units}) if proj.units else []
                amen_str = ", ".join(proj.amenities[:4]) or "—"
                address = proj.extra_props.get("address")
                location_str = address if address else f"{proj.neighbourhood}, {proj.city}"
                
                return (
                    f'<div class="project-card">'
                    f'<h4>{proj.project_name} <span class="{tag_c}">{tag_l}</span></h4>'
                    f'📍 {location_str}<br>'
                    f'🏗️ {proj.developer}<br>'
                    f'🏠 {", ".join(u_types) if u_types else "—"}<br>'
                    f'✨ {amen_str}'
                    f"</div>"
                )

            def get_minimal_row(proj, tag_c, tag_l):
                address = proj.extra_props.get("address")
                location_str = address if address else f"{proj.neighbourhood}, {proj.city}"
                return (
                    f'<div class="project-list-row">'
                    f'<div class="pl-title">{proj.project_name}</div>'
                    f'<div class="pl-loc">📍 {location_str}</div>'
                    f'<div class="pl-dev">🏗️ {proj.developer}</div>'
                    f"</div>"
                )

            # Sequentially render matched projects and reasoning
            for p in projects_data:
                p_name = p.get("project_name", "")
                reasoning = str(p.get("reasoning", "")).strip()
                
                # Robust matching: Try exact, then try substring either way
                matched_proj = None
                p_lower = p_name.lower().strip()
                for r in results:
                    r_lower = r.project_name.lower().strip()
                    if r_lower == p_lower or r_lower in p_lower or p_lower in r_lower:
                        matched_proj = r
                        break
                        
                # Even more robust matching: check if prominent words overlap
                if not matched_proj and p_lower:
                    p_words = set(p_lower.split())
                    for r in results:
                        r_words = set(r.project_name.lower().strip().split())
                        # If there is a strong word overlap (>50% of the shorter name's words match)
                        if len(p_words.intersection(r_words)) / max(1, min(len(p_words), len(r_words))) >= 0.5:
                            matched_proj = r
                            break
                
                html_rendered = ""
                if matched_proj:
                    tag_cls = {"both": "tag-both", "graph": "tag-graph", "vector": "tag-vector"}.get(
                        matched_proj.source, "tag-graph"
                    )
                    tag_label = {"both": "Graph + Vector", "graph": "Graph Match", "vector": "Vector Match"}.get(
                        matched_proj.source, matched_proj.source
                    )
                    
                    if is_large_list:
                        html_rendered = get_minimal_row(matched_proj, tag_cls, tag_label)
                    else:
                        html_rendered = get_project_card(matched_proj, tag_cls, tag_label)
                    
                    st.markdown(html_rendered, unsafe_allow_html=True)
                    time.sleep(0.02) # faster for lists
                
                if reasoning and not is_large_list:
                    # Render all within a single markdown call. Streamlit parses markdown inside HTML 
                    # if separated by blank lines
                    st.markdown(f'<div class="llm-explanation" markdown="1">\n\n{reasoning}\n\n</div>', unsafe_allow_html=True)
                    time.sleep(0.04)
                
                # We store generic "html_content" to handle both the card or the row seamlessly on reload
                p["html_content"] = html_rendered
                p["reasoning"] = reasoning # Store raw reasoning for history recreation
                
            if answer.get("conclusion"):
                st.markdown(answer.get("conclusion"))

            if "llm_metadata" in answer:
                meta = answer["llm_metadata"]
                st.markdown(f"<div style='font-size: 0.8rem; color: #718096; margin-top: 10px; text-align: right;'>🤖 Engine: {meta.get('model_name', 'unknown')} &nbsp;|&nbsp; 🪙 Tokens: {meta.get('total_tokens', 0)} ({meta.get('prompt_tokens', 0)} prompt + {meta.get('completion_tokens', 0)} completion)</div>", unsafe_allow_html=True)
        else:
            st.markdown(answer)

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
