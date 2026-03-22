"""
graphrag_answer.py — Token-efficient LLM relevance judge for vector-only results.

Design:
  - Graph results (source="graph" or "both") are ALWAYS trusted — Cypher already
    filtered them precisely. They are NEVER sent to the LLM judge.
  - Vector-only results (source="vector") are sent to a lightweight relevance judge.
  - The judge receives a compact but RICH per-project context:
      BHK options, unit types, areas, ALL amenities, special rooms, landmarks,
      and a truncated society description — enough to judge any query type.
  - Primary LLM: Groq (Llama 3.3 70b) — fast, free, generous limits.
  - Fallback LLM: Gemini 2.5 Flash — if Groq fails.
  - If both fail: all vector results pass (safe fallback, never drops results).
  - Judge returns strictly: {"results": [{"project_name": "...", "relevant": true/false}]}
    — no reasoning, no prose, minimal tokens.
  - The _fallback_answer() function builds the structured card data for the
    frontend. The LLM judge only gates which projects are included.
"""

import json
import re

import google.generativeai as genai
from groq import Groq
from loguru import logger

from graphrag_config import settings
from graphrag_intent import QueryIntent
from graphrag_retriever import ProjectResult


# ── Judge system prompt ────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
You are a relevance filter for a real estate search engine.

Given a user query and a list of residential project summaries, decide for each project
whether it is RELEVANT to the query.

First, classify the query type:
- SPECIFIC: user mentions concrete requirements (specific rooms, exact BHK, named amenities,
  facing direction, minimum area, developer name, etc.)
- GENERAL: broad or exploratory query (location only, lifestyle words, vague descriptions).

Relevance rules:
- SPECIFIC query: mark relevant=true if you feel the project satisfies around 75% or more of
  what the user asked for. Consider all stated requirements together and judge holistically —
  if the overall fit feels strong enough, mark true. If the project clearly falls short on
  most of the specific requirements, mark false.
- GENERAL query: mark relevant=true if the project has around 65% or more overlap with the
  user's description. Partial matches (right area, similar type, some amenities) count.
- When uncertain on a GENERAL query, prefer relevant=true.
- When uncertain on a SPECIFIC query, prefer relevant=false.
- Never drop a project solely on subjective/lifestyle mismatches (e.g. "luxury feel").
- Return ONLY raw JSON, no markdown, no explanation.

Output format (strict):
{"results": [{"project_name": "<exact name>", "relevant": true}, ...]}
"""


# ── Judge context builder — rich but token-efficient ──────────────────────────

def _build_judge_summary(p: ProjectResult) -> dict:
    """
    Build a compact but comprehensive per-project dict for the LLM judge.

    Covers ALL query dimensions:
      - BHK + unit types + areas         (for BHK / size queries)
      - Special rooms                    (for room-specific queries)
      - Full amenity list                (for amenity queries)
      - Landmarks                        (for proximity queries)
      - Truncated society description    (for fuzzy / lifestyle queries)

    Token budget: ~150-200 tokens per project at most.
    """
    # BHK numbers & unit types
    bhk_set = sorted({u.get("bhk") for u in p.units if u.get("bhk")})
    unit_types = sorted({u.get("unit_type") for u in p.units if u.get("unit_type")})

    # Area ranges (super built-up, carpet)
    sbua_vals = [u.get("super_builtup_sqft") for u in p.units if u.get("super_builtup_sqft")]
    carpet_vals = [u.get("carpet_sqft") for u in p.units if u.get("carpet_sqft")]
    area_parts = []
    if sbua_vals:
        lo, hi = min(sbua_vals), max(sbua_vals)
        area_parts.append(f"SBA {int(lo)}–{int(hi)} sqft" if lo != hi else f"SBA {int(lo)} sqft")
    if carpet_vals:
        lo, hi = min(carpet_vals), max(carpet_vals)
        area_parts.append(f"carpet {int(lo)}–{int(hi)} sqft" if lo != hi else f"carpet {int(lo)} sqft")

    # Special rooms — everything except generic bedroom/kitchen/hall/toilet/dining
    _COMMON_ROOMS = {"bedroom", "master bedroom", "kitchen", "hall", "toilet",
                     "bathroom", "wc", "dining", "passage", "lobby", "wash area"}
    special_rooms: set[str] = set()
    for u in p.units:
        for r in (u.get("rooms") or []):
            rname = (r.get("name") or "").strip()
            if rname and rname.lower() not in _COMMON_ROOMS:
                special_rooms.add(rname)
    # Also check project-level boolean flags for quick wins
    flag_rooms = []
    ep = p.extra_props
    if ep.get("has_pool") == 1:       flag_rooms.append("Swimming Pool")
    if ep.get("has_clubhouse") == 1:  flag_rooms.append("Clubhouse")
    if ep.get("has_park") == 1:       flag_rooms.append("Park/Garden")
    if ep.get("has_parking") == 1:    flag_rooms.append("Parking")

    # Society description — truncate hard at 150 chars to stay token-efficient
    soc_desc = (ep.get("society_description") or "").strip()
    soc_desc_short = (soc_desc[:150] + "…") if len(soc_desc) > 150 else soc_desc

    return {
        "project_name": p.project_name,
        "location": f"{p.neighbourhood}, {p.city}",
        "bhk": bhk_set,
        "unit_types": unit_types,
        "area": ", ".join(area_parts) if area_parts else None,
        "special_rooms": sorted(special_rooms)[:8] if special_rooms else None,
        "amenities": p.amenities or None,          # FULL list — names are short
        "nearby": p.landmarks[:6] if p.landmarks else None,
        "description": soc_desc_short or None,
    }


# ── Core judge function ────────────────────────────────────────────────────────

def _judge_vector_relevance(
    user_query: str,
    vector_only: list[ProjectResult],
    api_keys: dict,
) -> list[ProjectResult]:
    """
    Use Groq (Llama) → Gemini 2.5 Flash fallback to decide which vector-only
    projects are relevant to the user query.

    Returns the filtered list. On any LLM failure, returns all unchanged.
    """
    if not vector_only:
        return vector_only

    # Build compact summaries
    summaries = [_build_judge_summary(p) for p in vector_only]
    # Strip None values for cleaner JSON (saves tokens)
    summaries_clean = [{k: v for k, v in s.items() if v is not None} for s in summaries]

    judge_user_msg = (
        f"User query: {user_query}\n\n"
        f"Projects to evaluate:\n{json.dumps(summaries_clean, indent=2)}"
    )

    # ── Primary: Groq (Llama 3.3 70b) ──────────────────────────────────────
    groq_key = (api_keys or {}).get("GROQ_API_KEY") or settings.GROQ_API_KEY
    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": judge_user_msg},
                ],
                temperature=0.0,
                max_tokens=256,   # yes/no per project — very small output
            )
            raw = response.choices[0].message.content.strip()
            tokens_in  = response.usage.prompt_tokens if response.usage else "?"
            tokens_out = response.usage.completion_tokens if response.usage else "?"
            logger.info(f"[RelevanceJudge/Groq] tokens: {tokens_in} in / {tokens_out} out")
            return _parse_judge_response(raw, vector_only)
        except Exception as e:
            logger.warning(f"[RelevanceJudge] Groq failed ({e}). Trying Gemini fallback.")

    # ── Fallback: Gemini 2.5 Flash ─────────────────────────────────────────
    gemini_key = (api_keys or {}).get("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",   # always use 2.5 Flash for judge
                system_instruction=_JUDGE_SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            response = model.generate_content(judge_user_msg)
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            tokens = (response.usage_metadata.total_token_count
                      if hasattr(response, "usage_metadata") and response.usage_metadata else "?")
            logger.info(f"[RelevanceJudge/Gemini] tokens total: {tokens}")
            return _parse_judge_response(raw, vector_only)
        except Exception as e:
            logger.warning(f"[RelevanceJudge] Gemini also failed ({e}). Keeping all vector results.")

    # Both failed — safe fallback
    logger.warning("[RelevanceJudge] All LLMs failed. Returning vector results unfiltered.")
    return vector_only


def _parse_judge_response(
    raw: str,
    vector_only: list[ProjectResult],
) -> list[ProjectResult]:
    """Parse the LLM response and filter the vector-only list."""
    data = json.loads(raw)
    results_map: dict[str, bool] = {
        r["project_name"].lower(): bool(r["relevant"])
        for r in data.get("results", [])
        if isinstance(r, dict) and "project_name" in r and "relevant" in r
    }
    kept, dropped = [], []
    for p in vector_only:
        p_lower = p.project_name.lower()
        # Fuzzy match — LLM may slightly rephrase names
        match_key = next(
            (k for k in results_map if k in p_lower or p_lower in k), None
        )
        is_relevant = results_map.get(match_key, True)  # unknown → keep (safe)
        if is_relevant:
            kept.append(p)
        else:
            dropped.append(p.project_name)

    if dropped:
        logger.info(f"[RelevanceJudge] Filtered out {len(dropped)} vector-only result(s): {dropped}")
    logger.info(f"[RelevanceJudge] Kept {len(kept)} / {len(vector_only)} vector-only result(s)")
    return kept


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    context_text: str,
    project_results: list[ProjectResult],
    intent: QueryIntent,
    direct_answer_text: str = "",
    api_keys: dict = None,
) -> dict:
    """
    Filter vector-only results through the slim LLM relevance judge, then
    return the structured card data for the frontend.

    Graph/both results bypass the judge entirely.
    """
    if not project_results:
        ans = {
            "general_summary": "I couldn't find any projects matching your criteria. Try broadening your search.",
            "projects": [],
            "conclusion": "",
        }
        if direct_answer_text:
            ans["direct_answer"] = direct_answer_text
        return ans

    # ── Split by source ───────────────────────────────────────────────────────
    graph_and_both: list[ProjectResult] = []
    vector_only: list[ProjectResult] = []
    for r in project_results:
        (vector_only if r.source == "vector" else graph_and_both).append(r)

    logger.info(
        f"[generate_answer] {len(graph_and_both)} graph/both (always kept), "
        f"{len(vector_only)} vector-only (to judge)"
    )

    # ── Skip judge for global queries or when no vector-only results ──────────
    has_specific_filters = any([
        intent.bhk, intent.property_type, intent.amenities, intent.landmark_types,
        intent.specific_landmarks, intent.min_sqft, intent.max_sqft,
        intent.min_price_lakhs, intent.max_price_lakhs, intent.has_balcony,
        intent.has_parking, intent.entrance_facing, intent.developer,
        intent.project_names, intent.min_units_per_floor, intent.max_units_per_floor,
    ])
    is_global = any(
        ph in user_query.lower()
        for ph in ["show all", "list all", "all available", "what's available"]
    )
    # SKIP_VECTOR_JUDGE=true → bypass judge entirely; distance threshold already
    # gates quality. Zero judge tokens used. Set in .env to control behaviour.
    skip_judge = (
        (not vector_only)
        or (is_global and not has_specific_filters)
        or settings.SKIP_VECTOR_JUDGE
    )

    if skip_judge:
        reason = (
            "SKIP_VECTOR_JUDGE=true (distance-threshold gating only)"
            if settings.SKIP_VECTOR_JUDGE and vector_only
            else "no vector-only results or global query"
        )
        logger.info(f"[generate_answer] Skipping judge ({reason}).")
        judged_vector = vector_only
    else:
        judged_vector = _judge_vector_relevance(user_query, vector_only, api_keys or {})

    # ── Reassemble: graph/both first, then judged vector ─────────────────────
    final_results: list[ProjectResult] = graph_and_both + judged_vector

    if not final_results:
        ans = {
            "general_summary": "I couldn't find any projects that precisely match your requirements. Try adjusting your filters.",
            "projects": [],
            "conclusion": "",
        }
        if direct_answer_text:
            ans["direct_answer"] = direct_answer_text
        return ans

    ans = _fallback_answer(user_query, final_results)
    if direct_answer_text:
        ans["direct_answer"] = direct_answer_text
    return ans


# ── Structured card builder ────────────────────────────────────────────────────

def _fallback_answer(user_query: str, results: list[ProjectResult]) -> dict:
    """
    Build the structured project list rendered as cards on the frontend.
    All card content comes from the ProjectResult objects — no LLM prose.
    """
    projects = []
    for proj in results:
        lines: list[str] = []

        # Only add details that are NOT fully covered by the card headers/attributes above.
        
        # 1. Full Society Description
        soc_desc = proj.extra_props.get("society_description")
        if soc_desc and len(soc_desc) > 10:
            lines.append(f"**About:** *{soc_desc.strip()}*")

        projects.append({
            "project_name": proj.project_name,
            "reasoning": "\n\n".join(lines), 
        })

    return {
        "general_summary": f"Here are the residential projects matching your query: **{user_query}**\n",
        "projects": projects,
        "conclusion": "Hope this helps!",
    }
