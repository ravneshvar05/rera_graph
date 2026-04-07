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
import json
from pathlib import Path
import streamlit as st
from loguru import logger

from graphrag_cypher import generate_cypher
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
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        padding-bottom: 2rem !important; /* Add some breathing room at the bottom */
    }
    
    [data-testid="stChatInput"] {
        background-color: #1e2230 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 36px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
        padding: 6px 16px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(99, 179, 237, 0.6) !important;
        box-shadow: 0 8px 30px rgba(99, 179, 237, 0.3) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f7fafc !important;
        background-color: transparent !important;
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

    /* Expandable Compact Card (>10 results) */
    .project-compact {
        margin: 6px 0;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(30, 34, 45, 0.6);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .project-compact:hover {
        border-color: rgba(99, 179, 237, 0.25);
        box-shadow: 0 4px 14px rgba(99, 179, 237, 0.08);
    }
    .project-compact summary {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 18px;
        cursor: pointer;
        list-style: none;
        color: #e2e8f0;
        font-size: 0.95rem;
        transition: background 0.2s ease;
    }
    .project-compact summary::-webkit-details-marker { display: none; }
    .project-compact summary::after {
        content: '▸ Details';
        font-size: 0.75rem;
        color: #63b3ed;
        margin-left: 12px;
        white-space: nowrap;
        transition: transform 0.2s ease;
    }
    .project-compact[open] summary::after {
        content: '▾ Less';
    }
    .project-compact summary:hover {
        background: rgba(36, 41, 56, 0.8);
    }
    .project-compact .pc-name { font-weight: 600; color: #bee3f8; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .project-compact .pc-loc { flex: 1.5; color: #a0aec0; font-size: 0.88rem; padding: 0 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .project-compact .pc-tag { flex-shrink: 0; }
    .project-compact .pc-detail {
        padding: 0 18px 16px 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.7;
        animation: slideDown 0.2s ease-out;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .project-compact .pc-detail .pc-row {
        padding: 3px 0;
    }
    .project-compact .pc-detail .pc-rooms {
        margin-top: 6px;
        padding: 8px 12px;
        background: rgba(20, 24, 35, 0.5);
        border-radius: 8px;
        font-size: 0.85rem;
        color: #a0aec0;
    }

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

    /* ── GATE CLARIFICATION CARD ── */
    .gate-card {
        background: linear-gradient(135deg, rgba(99,179,237,0.07), rgba(159,122,234,0.04));
        border: 1px solid rgba(99, 179, 237, 0.25);
        border-left: 4px solid #63b3ed;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .gate-label {
        font-size: 0.72rem;
        color: #90cdf4;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .gate-question {
        font-size: 1.05rem;
        color: #e2e8f0;
        margin: 0;
        line-height: 1.55;
    }
    .gate-section-label {
        color: #90cdf4;
        font-weight: 600;
        margin: 14px 0 8px 0;
        font-size: 0.88rem;
    }
    .gate-opt-label {
        color: #a0aec0;
        font-weight: 500;
        margin: 14px 0 8px 0;
        font-size: 0.85rem;
    }
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
    enable_gate = st.toggle("🔍 Smart Clarification", value=True,
                            help="When ON: asks for city/BHK if your query is too broad, saving time and tokens.")

# ── Gate: available cities + helpers ─────────────────────────────────────────
AVAILABLE_CITIES = ["Ahmedabad", "Surat", "Vadodara", "Rajkot"]

_GATE_SYSTEM_PROMPT = """You are a query gate for a real estate search system in Gujarat, India.
The database has residential projects in: Ahmedabad, Surat, Vadodara, Rajkot.

Decide if the user query has enough specificity/location context to search directly.
You must return ONE of three states.

── STATE 1: ready=true ──
Return this if the query contains ANY of:
- A city name (Ahmedabad, Surat, Vadodara, Rajkot, or any Gujarat town/village)
- A specific locality, area, or neighbourhood (Bopal, Nikol, Satellite, Vinzol, Naroda, Sarkhej, Thaltej, Vastrapur, Chandkheda, Paldi, Gota, Naranpura, etc.)
- A road, highway, or street name (SG Highway, Sarkhej-Gandhinagar Highway, Ring Road, SP Ring Road, 132 feet road, Sindhu Bhavan Road, Bodakdev, etc.)
- A directional zone (west Ahmedabad, east side, south zone, north Gujarat, etc.)
- Any landmark, institution, or place name (near airport, Karnavati Club, near school, near hospital, near mall, SGVP, IIM, GIFT City, etc.)
- A specific project name or developer/builder name
- At least THREE independent non-location property filters (e.g. BHK + property type + amenity)
AND the BHK type IS mentioned explicitly in the query.
Output: {"ready": true}

── STATE 2: asked_for="bhk_optional" ──
Return this when:
- The query HAS location/project context (city, area, road, zone, landmark, project, etc.) OR has 3+ non-location filters
- BUT the BHK count is NOT mentioned (no "1 BHK", "2 BHK", "3 BHK", "2bhk", "two bedroom", etc.)
Output: {"ready": false, "asked_for": "bhk_optional", "question": "<short friendly question asking if they want to filter by BHK — mention it is optional>"}

── STATE 3: asked_for="city" ──
Return this ONLY when ALL of these are true simultaneously:
- The query has ZERO location context of any kind (no city, area, road, zone, landmark, project, developer, no "near X", no "in X")
- AND has fewer than 3 independent non-location property filters (e.g., just BHK and one amenity is not enough)
If the user mentions ANYTHING that sounds like a place or area — even vaguely — do NOT return this state.
Output: {"ready": false, "asked_for": "city", "question": "<friendly question asking which city from: Ahmedabad, Surat, Vadodara, Rajkot>"}

Examples:
- "find me 2bhk project with garden" → STATE 3 city  (zero location, only 2 filters: BHK + amenity)
- "apartments in Ahmedabad" → STATE 2 bhk_optional  (city present, BHK missing)
- "show all projects" → STATE 3 city  (zero location, zero filters)
- "3 BHK in Ahmedabad" → STATE 1 ready=true  (city + BHK both present)
- "2 BHK near SG highway" → STATE 1 ready=true  (road + BHK both present)
- "projects near Sarkhej" → STATE 2 bhk_optional  (area present, BHK missing)
- "near the school" → STATE 2 bhk_optional  (landmark present, BHK missing)
- "flat near SP ring road" → STATE 2 bhk_optional  (road present, BHK missing)
- "2 BHK penthouse with pool under 80 lakhs" → STATE 1 ready=true  (BHK + type + amenity + price = 4 non-location filters, ready without city)

Output ONLY valid JSON, nothing else — one of:
{"ready": true}
{"ready": false, "asked_for": "bhk_optional", "question": "..."}
{"ready": false, "asked_for": "city", "question": "..."}
"""


def _gate_llm_check(raw_query: str, api_keys: dict) -> dict | None:
    """Tiny LLM gate. Returns None to pass through, or {"asked_for": ..., "question": ...} to ask."""
    from graphrag_config import build_groq_pool, settings
    from groq import Groq
    import json

    groq_pool = build_groq_pool(api_keys)
    for _ in range(max(len(groq_pool), 1)):
        key = groq_pool.next()
        if not key:
            break
        try:
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _GATE_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            if data.get("ready", True):
                return None
            asked_for = data.get("asked_for", "city")
            default_q = (
                f"I have projects across {', '.join(AVAILABLE_CITIES)}! Which city are you looking in?"
                if asked_for == "city"
                else "Would you like to filter by BHK type? (completely optional — you can skip this)"
            )
            return {
                "asked_for": asked_for,
                "question": data.get("question", default_q),
            }
        except Exception as e:
            logger.warning(f"[Gate] LLM call failed ({e}). Passing through.")
    return None  # safe fallback: always pass through if gate fails


def _build_combined_query(original: str, city: str | None, bhk: int | None) -> str:
    """Build a natural combined query from original + user clarification selections."""
    orig = original.strip().rstrip(".")
    orig_lower = orig.lower()
    add_bhk = ""
    add_city = ""

    if bhk:
        if f"{bhk} bhk" not in orig_lower and f"{bhk}bhk" not in orig_lower.replace(" ", ""):
            add_bhk = f"{bhk} BHK"

    if city and city.lower() not in orig_lower:
        add_city = f"in {city}"

    if add_bhk and add_city:
        return f"{orig}, {add_bhk} {add_city}"
    elif add_bhk:
        return f"{orig}, {add_bhk}"
    elif add_city:
        return f"{orig} {add_city}"
    return orig


def _render_clarification_ui(pending: dict):
    """Render the clarification UI. Shows city+BHK chips (city missing) or BHK-only chips (bhk optional)."""
    question = pending.get("question", "Which city are you looking in?")
    asked_for = pending.get("asked_for", "city")
    selected_bhk = st.session_state.get("clarification_bhk")

    # ── BHK-only mode (city is already known, BHK is optional) ────────────────
    if asked_for == "bhk_optional":
        st.markdown(
            f'<div class="gate-card">'
            f'<div class="gate-label">🛏️ Optional filter</div>'
            f'<p class="gate-question">{question}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="gate-section-label">Select BHK type:</p>', unsafe_allow_html=True)
        bhk_cols = st.columns(6)
        for i, b in enumerate([1, 2, 3, 4, 5]):
            with bhk_cols[i]:
                is_sel = selected_bhk == b
                label = f"✓ {b} BHK" if is_sel else f"{b} BHK"
                if st.button(label, key=f"gate_bhk_only_{b}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    st.session_state["clarification_bhk"] = None if is_sel else b
                    st.rerun()
        with bhk_cols[5]:
            is_sel = selected_bhk == "Any"
            label = "✓ Any" if is_sel else "Any"
            if st.button(label, key="gate_bhk_only_any", use_container_width=True, type="primary" if is_sel else "secondary"):
                st.session_state["clarification_bhk"] = None if is_sel else "Any"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Search 🔍", key="search_bhk_only", type="primary", use_container_width=True):
            bhk_val = None if selected_bhk == "Any" else selected_bhk
            combined = _build_combined_query(pending["original_query"], None, bhk_val)
            st.session_state.messages.append({"role": "user", "content": combined})
            st.session_state["gate_resolved_query"] = combined
            del st.session_state["pending_clarification"]
            st.session_state.pop("clarification_bhk", None)
            st.rerun()
        return

    # ── City mode (city is missing — show city chips + optional BHK) ───────────
    st.markdown(
        f'<div class="gate-card">'
        f'<div class="gate-label">🔍 One quick question</div>'
        f'<p class="gate-question">{question}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    selected_city = st.session_state.get("clarification_city")

    # City chips (toggle)
    st.markdown('<p class="gate-section-label">📍 Select your city:</p>', unsafe_allow_html=True)
    city_cols = st.columns(len(AVAILABLE_CITIES) + 1)
    for i, city in enumerate(AVAILABLE_CITIES):
        with city_cols[i]:
            is_sel = selected_city == city
            label = f"✓ {city}" if is_sel else city
            if st.button(label, key=f"gate_city_{city}", use_container_width=True, type="primary" if is_sel else "secondary"):
                st.session_state["clarification_city"] = None if is_sel else city
                st.rerun()
    with city_cols[-1]:
        is_sel = selected_city == "All Cities"
        label = "✓ All Cities" if is_sel else "🌍 All Cities"
        if st.button(label, key="gate_city_all", use_container_width=True, type="primary" if is_sel else "secondary"):
            st.session_state["clarification_city"] = None if is_sel else "All Cities"
            st.rerun()

    # BHK optional chips (toggle)
    st.markdown('<p class="gate-opt-label">🛏️ BHK type <span style="color:#718096;font-size:0.8rem;">(optional)</span></p>', unsafe_allow_html=True)
    bhk_cols = st.columns(6)
    for i, b in enumerate([1, 2, 3, 4, 5]):
        with bhk_cols[i]:
            is_sel = selected_bhk == b
            label = f"✓ {b} BHK" if is_sel else f"{b} BHK"
            if st.button(label, key=f"gate_bhk_{b}", use_container_width=True, type="primary" if is_sel else "secondary"):
                st.session_state["clarification_bhk"] = None if is_sel else b
                st.rerun()
    with bhk_cols[5]:
        is_sel = selected_bhk == "Any"
        label = "✓ Any" if is_sel else "Any"
        if st.button(label, key="gate_bhk_any_city", use_container_width=True, type="primary" if is_sel else "secondary"):
            st.session_state["clarification_bhk"] = None if is_sel else "Any"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Search 🔍", key="search_city_mode", type="primary", disabled=(not selected_city), use_container_width=True):
        city_val = None if selected_city == "All Cities" else selected_city
        bhk_val = None if selected_bhk == "Any" else selected_bhk
        combined = _build_combined_query(pending["original_query"], city_val, bhk_val)
        st.session_state.messages.append({"role": "user", "content": combined})
        st.session_state["gate_resolved_query"] = combined
        del st.session_state["pending_clarification"]
        st.session_state.pop("clarification_bhk", None)
        st.session_state.pop("clarification_city", None)
        st.rerun()



# ── Session state ──────────────────────────────────────────────────────────────
SETTINGS_PATH = Path("user_settings.json")

if "user_settings" not in st.session_state:
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r") as f:
                st.session_state.user_settings = json.load(f)
        except Exception:
            st.session_state.user_settings = {}
    else:
        st.session_state.user_settings = {}

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

# ── Sidebar Bottom Settings ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**⚙️ API Settings**")
    st.markdown("<div style='font-size: 0.8rem; color: #a0aec0; margin-bottom: 10px;'>Keys are stored locally.</div>", unsafe_allow_html=True)
    
    def save_settings():
        with open(SETTINGS_PATH, "w") as f:
            json.dump(st.session_state.user_settings, f)

    # ── Migrate old single-key format to new multi-key format ─────────────────
    # If the user had a single GEMINI_API_KEY saved before, pre-fill the new pool box
    existing_gemini = st.session_state.user_settings.get("GEMINI_API_KEYS", "")
    if not existing_gemini and st.session_state.user_settings.get("GEMINI_API_KEY", ""):
        existing_gemini = st.session_state.user_settings["GEMINI_API_KEY"]

    existing_groq = st.session_state.user_settings.get("GROQ_API_KEYS", "")
    if not existing_groq and st.session_state.user_settings.get("GROQ_API_KEY", ""):
        existing_groq = st.session_state.user_settings["GROQ_API_KEY"]

    st.markdown("<div style='font-size: 0.78rem; color: #a0aec0; margin-bottom: 6px;'>Enter one key per line. Multiple keys are used in round-robin — if one hits the limit, the next is used automatically.</div>", unsafe_allow_html=True)

    gemini_keys_raw = st.text_area(
        "🔵 Gemini API Keys (cypher generation)",
        value=existing_gemini,
        height=110,
        placeholder="AIzaSy...key1\nAIzaSy...key2\nAIzaSy...key3\nAIzaSy...key4",
        key="gemini_keys_input",
    )
    if gemini_keys_raw != st.session_state.user_settings.get("GEMINI_API_KEYS", ""):
        st.session_state.user_settings["GEMINI_API_KEYS"] = gemini_keys_raw
        save_settings()

    groq_keys_raw = st.text_area(
        "🟠 Groq API Keys (relevance judge)",
        value=existing_groq,
        height=80,
        placeholder="gsk_...key1\ngsk_...key2",
        key="groq_keys_input",
    )
    if groq_keys_raw != st.session_state.user_settings.get("GROQ_API_KEYS", ""):
        st.session_state.user_settings["GROQ_API_KEYS"] = groq_keys_raw
        save_settings()

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
            # Direct answer block (for LOOKUP/AGGREGATE queries)
            if content.get("direct_answer"):
                st.info(content["direct_answer"])

            st.markdown(content.get("general_summary", ""))
            
            projects_data = content.get("projects", [])
            is_large_list = len(projects_data) > 10
            
            for p in projects_data:
                if p.get("html_content"):
                    st.markdown(p["html_content"], unsafe_allow_html=True)
                # reasoning text not shown — all info is rendered inside the card HTML above
                    
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

# ── Pending Clarification UI (rendered after chat history if gate fired) ────────
if "pending_clarification" in st.session_state:
    with st.chat_message("assistant"):
        _render_clarification_ui(st.session_state["pending_clarification"])

# ── Input ──────────────────────────────────────────────────────────────────────
# Handle preset query from sidebar buttons
preset = st.session_state.pop("preset_query", None)
# gate_resolved_query is set when user picks city/BHK from the clarification UI
resolved = st.session_state.pop("gate_resolved_query", None)
user_input = st.chat_input("Describe the property you're looking for…")

# If user types something fresh while a clarification is pending, treat it as a new query
if user_input and st.session_state.get("pending_clarification"):
    del st.session_state["pending_clarification"]
    st.session_state.pop("clarification_bhk", None)
    st.session_state.pop("clarification_city", None)

# Determine active query (resolved > preset > typed; skip typed if still pending clarification)
if resolved:
    query = resolved
elif preset:
    query = preset
elif user_input and not st.session_state.get("pending_clarification"):
    query = user_input
else:
    query = None

if query:
    # Only append to messages if this is a fresh typed/preset query (resolved was already appended
    # inside _render_clarification_ui when the city chip was clicked)
    if not resolved:
        st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):

        # ── Gate check (only for fresh typed queries; skip for preset/resolved) ──
        if enable_gate and not resolved and not preset:
            with st.status("🔍 Checking query…", expanded=False) as gate_status:
                gate_result = _gate_llm_check(query, st.session_state.user_settings)
            if gate_result:
                # Gate fired — remove the user message we just added (will be re-added
                # as the combined query after user picks city)
                if st.session_state.messages and st.session_state.messages[-1]["content"] == query:
                    st.session_state.messages.pop()
                st.session_state["pending_clarification"] = {
                    "original_query": query,
                    "question": gate_result["question"],
                    "asked_for": gate_result.get("asked_for", "city"),
                }
                _render_clarification_ui(st.session_state["pending_clarification"])
                st.stop()  # don't run the main pipeline

        with st.status("Searching properties…", expanded=True) as status:
            # Step 1: Generate Cypher + Intent in a SINGLE LLM call
            status.write("🧠 Understanding your query & generating graph query…")
            cypher_result = generate_cypher(query, api_keys=st.session_state.user_settings)
            intent = cypher_result.intent

            # Step 2: Retrieve (Graph + Vector run in PARALLEL internally)
            status.write("🔍 Searching knowledge graph and vector index…")
            retriever: DualRetriever = st.session_state.retriever
            context_text, results, direct_answer_text = retriever.retrieve_and_assemble(
                intent, query, cypher_result=cypher_result
            )

            # Step 3: Generate answer
            status.write("✍️ Generating recommendations…")
            answer = generate_answer(query, context_text, results, intent, direct_answer_text, api_keys=st.session_state.user_settings)

            final_count = len(answer.get("projects", [])) if isinstance(answer, dict) else len(results)
            status.update(
                label=f"✅ Found {final_count} project(s)", state="complete", expanded=False
            )

        # ── Display answer ─────────────────────────────────────────────────────
        if isinstance(answer, dict):
            # Direct answer block (LOOKUP / AGGREGATE queries)
            if answer.get("direct_answer"):
                st.info(answer["direct_answer"])

            st.markdown(answer.get("general_summary", ""))
            
            projects_data = answer.get("projects", [])
            is_large_list = len(projects_data) > 10
            
            def get_project_card(proj, tag_c, tag_l):
                u_types = list({u.get("unit_type", "?") for u in proj.units}) if proj.units else []
                amen_str = ", ".join(proj.amenities[:6]) or "—"
                address = proj.extra_props.get("address")
                location_str = address if address else f"{proj.neighbourhood}, {proj.city}"

                # BHK — deduplicated, sorted
                bhk_vals = sorted({u.get("bhk") for u in proj.units if u.get("bhk")})
                bhk_str = ", ".join(f"{b} BHK" for b in bhk_vals) if bhk_vals else ""

                # Build extra info lines
                extra_lines = []
                status_val = proj.extra_props.get("project_status")
                if status_val and status_val.upper() not in ("UNKNOWN", "NONE"):
                    status_display = status_val.replace("_", " ").title()
                    extra_lines.append(f'📋 Status: {status_display}')
                poss_date = proj.extra_props.get("possession_date")
                if poss_date:
                    extra_lines.append(f'📅 Possession: {poss_date}')
                rera = proj.extra_props.get("rera_number")
                if rera:
                    extra_lines.append(f'🔖 RERA: {rera}')
                bldgs = proj.extra_props.get("total_buildings")
                if bldgs:
                    extra_lines.append(f'🏢 Buildings: {bldgs}')

                # Show key rooms from first unit
                room_snippets = []
                if proj.units:
                    for u in proj.units[:2]:  # first 2 unit types
                        rooms = u.get("rooms") or []
                        for r in rooms[:4]:  # first 4 rooms per unit
                            rname = r.get("name", "")
                            rarea = r.get("area_sqft")
                            if rname and rarea:
                                try:
                                    room_snippets.append(f"{rname}: {float(rarea):.0f} sqft")
                                except (ValueError, TypeError):
                                    room_snippets.append(f"{rname}: {rarea} sqft")
                        if room_snippets:
                            break  # only show rooms from one unit type

                room_line = ""
                if room_snippets:
                    room_line = f'<br>📐 {" · ".join(room_snippets[:4])}'

                extra_html = ""
                if extra_lines:
                    extra_html = '<br>' + ' &nbsp;|&nbsp; '.join(extra_lines)

                return (
                    f'<div class="project-card">'
                    f'<h4>{proj.project_name} <span class="{tag_c}">{tag_l}</span></h4>'
                    f'📍 {location_str}<br>'
                    f'🏗️ {proj.developer}<br>'
                    + (f'🛏️ {bhk_str}<br>' if bhk_str else '')
                    + f'🏠 {", ".join(u_types) if u_types else "—"}<br>'
                    f'✨ {amen_str}'
                    f'{room_line}'
                    f'{extra_html}'
                    f"</div>"
                )

            def get_expandable_card(proj, tag_c, tag_l):
                """Compact expandable card for large result sets (>10 projects)."""
                address = proj.extra_props.get("address")
                location_str = address if address else f"{proj.neighbourhood}, {proj.city}"

                # BHK — deduplicated, sorted
                bhk_vals = sorted({u.get("bhk") for u in proj.units if u.get("bhk")})
                bhk_str = ", ".join(f"{b} BHK" for b in bhk_vals) if bhk_vals else ""

                # Build detail lines for the expandable section
                detail_lines = []
                detail_lines.append(f'<div class="pc-row">🏗️ <strong>Developer:</strong> {proj.developer}</div>')

                if bhk_str:
                    detail_lines.append(f'<div class="pc-row">🛏️ <strong>BHK:</strong> {bhk_str}</div>')

                u_types = list({u.get("unit_type", "?") for u in proj.units}) if proj.units else []
                if u_types:
                    detail_lines.append(f'<div class="pc-row">🏠 <strong>Units:</strong> {", ".join(u_types)}</div>')

                if proj.amenities:
                    amen_str = ", ".join(proj.amenities[:8])
                    detail_lines.append(f'<div class="pc-row">✨ <strong>Amenities:</strong> {amen_str}</div>')

                # Status / Possession / RERA
                status_val = proj.extra_props.get("project_status")
                if status_val and status_val.upper() not in ("UNKNOWN", "NONE", ""):
                    detail_lines.append(f'<div class="pc-row">📋 <strong>Status:</strong> {status_val.replace("_", " ").title()}</div>')
                poss_date = proj.extra_props.get("possession_date")
                if poss_date:
                    detail_lines.append(f'<div class="pc-row">📅 <strong>Possession:</strong> {poss_date}</div>')
                rera = proj.extra_props.get("rera_number")
                if rera:
                    detail_lines.append(f'<div class="pc-row">🔖 <strong>RERA:</strong> {rera}</div>')
                bldgs = proj.extra_props.get("total_buildings")
                if bldgs:
                    detail_lines.append(f'<div class="pc-row">🏢 <strong>Buildings:</strong> {bldgs}</div>')

                # Room info from first unit
                room_snippets = []
                if proj.units:
                    for u in proj.units[:2]:
                        rooms = u.get("rooms") or []
                        for r in rooms[:4]:
                            rname = r.get("name", "")
                            rarea = r.get("area_sqft")
                            if rname and rarea:
                                try:
                                    room_snippets.append(f"{rname}: {float(rarea):.0f} sqft")
                                except (ValueError, TypeError):
                                    room_snippets.append(f"{rname}: {rarea} sqft")
                        if room_snippets:
                            break
                if room_snippets:
                    detail_lines.append(f'<div class="pc-rooms">📐 {" · ".join(room_snippets[:4])}</div>')

                # Society snippet — only show if the description is complete (not truncated)
                soc_desc = proj.extra_props.get("society_description", "")
                if soc_desc and len(soc_desc) > 10 and len(soc_desc) <= 300:
                    detail_lines.append(f'<div class="pc-row" style="color:#718096;font-size:0.85rem;margin-top:4px">{soc_desc}</div>')

                if proj.landmarks:
                    lm_str = ", ".join(proj.landmarks[:5])
                    detail_lines.append(f'<div class="pc-row">🗺️ <strong>Nearby:</strong> {lm_str}</div>')

                detail_html = "\n".join(detail_lines)

                return (
                    f'<details class="project-compact">'
                    f'<summary>'
                    f'<span class="pc-name">{proj.project_name}</span>'
                    f'<span class="pc-loc">📍 {location_str}</span>'
                    f'<span class="pc-tag {tag_c}">{tag_l}</span>'
                    f'</summary>'
                    f'<div class="pc-detail">{detail_html}</div>'
                    f'</details>'
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
                        html_rendered = get_expandable_card(matched_proj, tag_cls, tag_label)
                    else:
                        html_rendered = get_project_card(matched_proj, tag_cls, tag_label)
                    
                    st.markdown(html_rendered, unsafe_allow_html=True)
                
                if reasoning and not is_large_list:
                    # Render all within a single markdown call. Streamlit parses markdown inside HTML 
                    # if separated by blank lines
                    st.markdown(f'<div class="llm-explanation" markdown="1">\n\n{reasoning}\n\n</div>', unsafe_allow_html=True)
                
                # We store generic "html_content" to handle both the card or the row seamlessly on reload
                p["html_content"] = html_rendered
                p["reasoning"] = reasoning # Store raw reasoning for history recreation
                
            if answer.get("conclusion"):
                st.markdown(answer.get("conclusion"))

            # Display token usage for the active model (Commented out per user request)
            # st.markdown(f"<div style='font-size: 0.8rem; color: #718096; margin-top: 10px; text-align: right;'>🤖 Cypher Engine tokens used: <strong>{cypher_result.tokens_used}</strong> | Engine: <strong>{cypher_result.engine_used}</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown(answer)
            # st.markdown(f"<div style='font-size: 0.8rem; color: #718096; margin-top: 10px; text-align: right;'>🤖 Cypher Engine tokens used: <strong>{cypher_result.tokens_used}</strong> | Engine: <strong>{cypher_result.engine_used}</strong></div>", unsafe_allow_html=True)

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
