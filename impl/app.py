"""
app.py — Premium Streamlit chatbot UI for the RERA Real Estate Project Finder.

Run with:
    streamlit run app.py
"""

import streamlit as st
from config import settings
from logger import logger
from query_planner import plan_query
from search import search
from answer_generator import generate_answer

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* App background */
  .stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    min-height: 100vh;
  }

  /* Header */
  .chat-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.3);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    text-align: center;
  }
  .chat-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
  }
  .chat-header p {
    color: #94a3b8;
    margin: 0.6rem 0 0;
    font-size: 1.05rem;
    font-weight: 400;
  }

  /* Animations */
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Chat bubbles */
  .user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 0.9rem 1.3rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 6rem;
    font-size: 1rem;
    line-height: 1.5;
    box-shadow: 0 8px 30px rgba(99,102,241,0.3);
    animation: slideUp 0.35s ease forwards;
  }
  .bot-bubble {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2);
    color: #e2e8f0;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 6rem 0.5rem 0;
    font-size: 1rem;
    line-height: 1.6;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    animation: slideUp 0.35s ease forwards;
  }
  .welcome-bubble {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2);
    color: #e2e8f0;
    padding: 1.5rem 2rem;
    border-radius: 20px;
    margin: 0.5rem 0 1.5rem 0;
    font-size: 1.05rem;
    line-height: 1.6;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    animation: slideUp 0.35s ease forwards;
  }

  /* Query type badge */
  .qtype-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .qtype-SEARCH    { background: rgba(99,102,241,0.2);  color: #818cf8; border: 1px solid rgba(99,102,241,0.4); }
  .qtype-AGGREGATE { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
  .qtype-COMPARE   { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
  .qtype-DETAIL    { background: rgba(139,92,246,0.15); color: #c084fc; border: 1px solid rgba(139,92,246,0.3); }

  /* Project cards */
  .project-card {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    color: #e2e8f0;
    transition: all 0.25s ease;
  }
  .project-card:hover {
    border-color: rgba(139,92,246,0.5);
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(99,102,241,0.15);
  }
  .proj-header { display: flex; justify-content: space-between; align-items: flex-start; }
  .proj-title  { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; }
  .proj-dev    { font-size: 0.9rem; color: #94a3b8; margin-top: 2px; }
  .proj-loc    { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
  .score-badge {
    font-size: 0.75rem; font-weight: 700;
    background: rgba(99,102,241,0.2);
    color: #818cf8;
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid rgba(99,102,241,0.3);
    white-space: nowrap;
  }
  .proj-tags { margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 6px; }
  .tag {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.2);
  }

  /* Unit list inside card */
  .units-section {
    margin-top: 1rem;
    border-top: 1px solid rgba(99,102,241,0.15);
    padding-top: 0.8rem;
  }
  .units-label {
    font-size: 0.72rem; font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
  }
  .unit-row {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin: 4px 0;
    padding: 6px 10px;
    background: rgba(99,102,241,0.06);
    border-radius: 8px;
    border-left: 3px solid rgba(99,102,241,0.3);
  }
  .unit-feat {
    color: #a5b4fc;
    font-size: 0.78rem;
    margin-left: 6px;
  }

  /* Input */
  [data-testid="stChatInput"] {
    background: transparent !important;
  }
  .stChatInputContainer {
    background: #1e293b !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 24px !important;
  }
  .stChatInputContainer:focus-within {
    border-color: rgba(139,92,246,0.8) !important;
    box-shadow: 0 0 15px rgba(139,92,246,0.2) !important;
  }
  .stChatInputContainer textarea {
    color: #f1f5f9 !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Buttons */
  .stButton > button {
    border-radius: 20px !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    background: rgba(99,102,241,0.1) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    height: 80px !important;
    width: 100% !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  .stButton > button:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(99,102,241,0.7) !important;
    transform: translateY(-2px) !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer { visibility: hidden !important; }
  header { background: transparent !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []     # {role, content} for LLM context

init_state()


# ─── Card Renderer ────────────────────────────────────────────────────────────

def render_project_card(p: dict) -> str:
    title = p.get("project_name", "Project")
    dev = p.get("developer_name", "")
    city = p.get("city", "")
    nbhd = p.get("neighbourhood", "")
    loc = f"📍 {city}" + (f", {nbhd}" if nbhd else "")

    score = p.get("relevance_score", 0)
    status = p.get("project_status", "")
    possession = p.get("possession_date", "")

    # Project-level amenity tags
    tags = []
    if p.get("has_clubhouse"):     tags.append("🏛️ Clubhouse")
    if p.get("has_pool"):          tags.append("🏊 Pool")
    if p.get("has_park"):          tags.append("🌳 Park")
    if p.get("has_sports_courts"): tags.append("🎾 Sports")
    if p.get("has_parking"):       tags.append("🚗 Parking")

    tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)

    status_line = ""
    if status and status != "UNKNOWN":
        status_line = f' · <span style="color:#34d399">{status}</span>'
    if possession:
        status_line += f' · Possession: {possession}'

    # Units section
    units = p.get("matching_units", [])
    units_html = ""
    if units:
        units_html = '<div class="units-section"><div class="units-label">Matching Configurations</div>'
        for u in units[:6]:
            uline = f"<b>{u.get('bhk')} BHK {u.get('property_type')}</b>"
            if u.get("super_builtup_sqft"):
                uline += f" &nbsp;·&nbsp; {u['super_builtup_sqft']} sqft"
            elif u.get("carpet_area_sqft"):
                uline += f" &nbsp;·&nbsp; carpet {u['carpet_area_sqft']} sqft"
            if u.get("entrance_facing"):
                uline += f" &nbsp;·&nbsp; {u['entrance_facing']}-facing"
            feats = []
            for col, lbl in [
                ("has_pooja_room","🛕"), ("has_study_room","📚"),
                ("has_terrace","🌿"), ("has_garden","🪴"),
                ("has_home_theatre","🎬"), ("has_gym","💪"),
                ("has_courtyard","🏡"), ("has_dressing_room","👗"),
            ]:
                if u.get(col):
                    feats.append(lbl)
            feat_html = f'<span class="unit-feat">{" ".join(feats)}</span>' if feats else ""
            units_html += f'<div class="unit-row">✓ {uline}{feat_html}</div>'
        units_html += "</div>"

    return f"""
<div class="project-card">
  <div class="proj-header">
    <div>
      <div class="proj-title">{title}</div>
      <div class="proj-dev">by {dev}</div>
      <div class="proj-loc">{loc}{status_line}</div>
    </div>
    <div class="score-badge">⭐ {score:.0%}</div>
  </div>
  <div class="proj-tags">{tags_html}</div>
  {units_html}
</div>"""


# ─── Message Renderer ─────────────────────────────────────────────────────────

def render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        qt = msg.get("query_type", "SEARCH")
        badge = f'<span class="qtype-badge qtype-{qt}">{qt}</span><br>'
        st.markdown(
            f'<div class="bot-bubble">{badge}{content}</div>',
            unsafe_allow_html=True,
        )
        for proj in msg.get("results", []):
            st.markdown(render_project_card(proj), unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="chat-header">
  <h1>🏠 RERA Project Finder</h1>
  <p>Describe your dream home — I'll find the best matches across all projects.</p>
</div>
""", unsafe_allow_html=True)


# ─── Chat History ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    render_message(msg)


# ─── Welcome Screen ───────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
<div class="welcome-bubble">
  👋 Hello! I can search across all real estate projects from RERA brochures.<br><br>
  I understand all kinds of queries:<br>
  &nbsp;&nbsp;• <b>Search</b>: "3 BHK villa with home theatre and garden in Ahmedabad"<br>
  &nbsp;&nbsp;• <b>Multi-city</b>: "2 BHK in Ahmedabad or Surat with pool"<br>
  &nbsp;&nbsp;• <b>Near landmarks</b>: "Projects near Karnavati Club"<br>
  &nbsp;&nbsp;• <b>Aggregations</b>: "How many villas with pooja room are there?"<br>
  &nbsp;&nbsp;• <b>Compare</b>: "Compare OUM ORBIT and KP Villas"<br>
  &nbsp;&nbsp;• <b>Deep details</b>: "Full details of KP Villas with all room sizes"
</div>
""", unsafe_allow_html=True)

    suggestions = [
        "3 BHK Villa with garden in Ahmedabad",
        "2 BHK apartment with gym and pool",
        "How many projects have a clubhouse?",
        "Compare OUM ORBIT and KP Villas",
        "Projects near Karnavati Club",
        "Full details of KP Villas",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state["_chip"] = s
            st.rerun()


# ─── Input ────────────────────────────────────────────────────────────────────

chip = st.session_state.pop("_chip", None)
user_input = st.chat_input(placeholder="Describe the home you're looking for...")
active_query = user_input.strip() if user_input else chip


# ─── Process Query ────────────────────────────────────────────────────────────

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    st.session_state.history.append({"role": "user", "content": active_query})
    render_message({"role": "user", "content": active_query})

    with st.spinner("🔍 Searching projects..."):
        try:
            # Step 1: Plan query
            plan = plan_query(active_query, st.session_state.history)

            if plan.needs_clarification:
                bot_reply = plan.clarification_question
                result_payload = {"query_type": "SEARCH", "results": [], "aggregate": None}
            else:
                # Steps 2-4: Search
                result_payload = search(plan, top_k=settings.DEFAULT_TOP_K)

                # Step 5: Generate answer
                bot_reply = generate_answer(
                    user_query=active_query,
                    search_result=result_payload,
                    conversation_history=st.session_state.history,
                )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            bot_reply = "Something went wrong. Please try again!"
            result_payload = {"query_type": "SEARCH", "results": [], "aggregate": None}

    qt = result_payload.get("query_type", "SEARCH")
    results = result_payload.get("results", [])

    bot_msg = {
        "role": "assistant",
        "content": bot_reply,
        "query_type": qt,
        "results": results,
    }
    st.session_state.messages.append(bot_msg)
    st.session_state.history.append({"role": "assistant", "content": bot_reply})
    st.rerun()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏠 RERA Project Finder")
    st.caption("Powered by Gemini · SQLite · ChromaDB")
    st.divider()

    st.markdown("**💡 Query Examples**")
    tips = [
        "3 BHK Villa in Ahmedabad with pooja room",
        "Apartments with home theatre or study room",
        "Projects near Karnavati Club",
        "How many 2 BHK projects are there?",
        "Compare OUM ORBIT and KP Villas",
        "Full details of KP Villas",
        "East-facing 4 BHK villa with garden",
        "2 BHK in Ahmedabad or Surat",
    ]
    for tip in tips:
        st.markdown(f"<small style='color:#64748b'>→ {tip}</small>", unsafe_allow_html=True)

    st.divider()

    # Quick stats
    try:
        import sqlite3
        from schema import get_connection as _gc
        c = _gc()
        n_proj = c.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        n_units = c.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        n_rooms = c.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
        c.close()
        st.markdown(f"**📊 Database**")
        col1, col2 = st.columns(2)
        col1.metric("Projects", n_proj)
        col2.metric("Units", n_units)
        st.metric("Room records", n_rooms)
    except Exception:
        pass

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
