"""
graphrag_answer.py — LLM-powered answer generator.

Takes the assembled context from graphrag_retriever.py and the original user
query, and generates a natural language recommendation response.

The LLM acts as the "synthesis" layer of GraphRAG:
  graph context + vector context → structured LLM prompt → human answer
"""

import json
from groq import Groq
from loguru import logger

from graphrag_config import settings
from graphrag_intent import QueryIntent
from graphrag_retriever import ProjectResult

# ── Configure Groq client ───────────────────────────────────────────────────
_groq_client = Groq(api_key=settings.GROQ_API_KEY)


# ── System prompt ─────────────────────────────────────────────────────────────

_ANSWER_SYSTEM_PROMPT = """You are a knowledgeable and helpful real estate advisor specializing in residential properties in Gujarat, India (primarily Ahmedabad and Surat).

Your role is to help users find the right residential project based on their needs.

You will be given:
1. The user's original query
2. A list of retrieved residential projects with their details (extracted from actual project brochures)

Your task:
- Recommend the most suitable projects based on the user's stated requirements
- If multiple projects match, detail them in order of relevance

You MUST respond in strict JSON format. Do not use Markdown backticks for the JSON block, just raw JSON.
The JSON must have the following exact structure:
{
  "general_summary": "A friendly introductory sentence or two.",
  "projects": [
    {
      "project_name": "Exact Name of the project from context",
      "reasoning": "Brief Markdown explanation (MAX 2 bullet points) of why this project is recommended and key details. IMPORTANT: If the user asks about floor layouts or units per floor, you MUST explicitly state the 'units per floor' in this reasoning block based on the provided context. You MUST use dashes (-) for bullet points, and separate each bullet point with a newline character (\\n) so that the entire reasoning is returned as a single valid JSON string. Do NOT use asterisks (*). IF THERE ARE MORE THAN 20 PROJECTS TOTAL, LEAVE THIS STRING EMPTY \"\"."
    }
  ],
  "conclusion": "A brief summary or next steps suggestion."
}

Important rules:
- IF THERE ARE MORE THAN 20 PROJECTS TO RECOMMEND, DO NOT WRITE ANY REASONING. JUST PROVIDE THE PROJECT NAMES AND A GOOD GENERAL SUMMARY.
- ONLY use information from the provided context — do not invent data
- If price/area is not in the context, do not guess it
- Always ground your recommendations in the actual retrieved data
- The project_name MUST perfectly match the name provided in the context.
- CRITICAL: You MUST filter out and omit any projects from the context that do not meaningfully match the user's specific requirements (e.g. if they asked for a 1 BHK, do not list a 3 BHK, if they asked for a clubhouse, do not list projects without a clubhouse). 
- STRICT LOGIC: Pay very close attention to AND vs OR conditions in the user's query. 
  - If a user asks for "Location A OR Location B AND Amenity C", a project is ONLY a perfect match if it actually HAS Amenity C AND is in either Location A OR Location B.
  - PERFECT MATCHES: Only projects that satisfy ALL strict mandatory criteria should be included in the 'projects' array (these will be rendered as detailed UI cards).
  - PARTIAL MATCHES: If a project only meets some conditions but fails others (e.g. it is in the right location but lacks the requested amenity), DO NOT include it in the 'projects' array. Instead, briefly mention these partial matches in the 'conclusion' string as plain text (e.g. "Note: Project X is in Location A but lacks Amenity C...").
- If no projects match perfectly, leave the 'projects' array empty and explain the partial matches in the 'general_summary' or 'conclusion'.
"""


# ── Answer generator ──────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    context_text: str,
    project_results: list[ProjectResult],
    intent: QueryIntent,
    direct_answer_text: str = "",
    api_keys: dict = None,
) -> dict:
    """
    Generate a JSON-structured recommendation answer.

    Args:
        user_query: The original user query string.
        context_text: The assembled context from DualRetriever.
        project_results: Structured project results (for fallback display).
        intent: The structured parser intent.
        direct_answer_text: Pre-formatted direct answer for LOOKUP/AGGREGATE queries.

    Returns:
        A formatted dictionary answer suitable for display in the chat UI.
        Includes 'direct_answer' key if direct_answer_text is provided.
    """
    if not project_results:
        ans = {
            "general_summary": "I couldn't find any projects matching your criteria in our current database. Try broadening your search.",
            "projects": [],
            "conclusion": ""
        }
        if direct_answer_text:
            ans["direct_answer"] = direct_answer_text
        return ans

    # Optimization: bypass LLM to prevent long inference times ONLY for genuine global/overview queries.
    # We check if there are ANY specific filters applied in the intent.
    has_specific_filters = any([
        intent.bhk, intent.property_type, intent.amenities, intent.landmark_types,
        intent.specific_landmarks, intent.min_sqft, intent.max_sqft,
        intent.min_price_lakhs, intent.max_price_lakhs, intent.has_balcony,
        intent.has_parking, intent.entrance_facing, intent.developer,
        intent.project_names, intent.min_units_per_floor, intent.max_units_per_floor
    ])

    is_global_command = any(ph in user_query.lower() for ph in ["show all", "list all", "all available", "what's available"])
    
    # Bypass ONLY if it's a generic "list all" without ANY specific criteria (amenities, BHK, etc.)
    if is_global_command and not has_specific_filters and len(project_results) > 10:
        logger.info("Bypassing LLM generation for genuine filter-free global query to save time.")
        return _fallback_answer(user_query, project_results)

    # Prevent rate limit errors by hard-capping the context size
    max_chars = 35000
    if len(context_text) > max_chars:
        logger.warning(f"Context too long ({len(context_text)} chars). Truncating to {max_chars} chars.")
        context_text = context_text[:max_chars] + "\n\n...[Context truncated due to size limits]..."

    # ────────── TEMPORARILY DISABLED LLM FOR GRAPH DEBUGGING ──────────
    # Bypass LLM generation entirely and just return the structured fallback answer
    # so we can see exactly what the Graph DB retrieved without hitting rate limits.
    logger.info("TEMPORARY: Bypassing LLM generation to debug Graph Retrieval.")
    ans = _fallback_answer(user_query, project_results)
    if direct_answer_text:
        ans["direct_answer"] = direct_answer_text
    return ans
    # ──────────────────────────────────────────────────────────────────
    
    # ... (Original code commented out or bypassed)
    prompt = f"""{_ANSWER_SYSTEM_PROMPT}

---

{context_text}

---

Based on the above retrieved project data, please provide a clear, helpful recommendation to the user.

User Query: {user_query}
"""

    try:
        response = _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"{context_text}\n\nBased on the above retrieved project data, please provide a clear, helpful recommendation to the user.\n\nUser Query: {user_query}"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        answer_text = response.choices[0].message.content.strip()
        logger.success(f"Answer generated ({len(answer_text)} chars)")
        
        try:
            parsed = json.loads(answer_text)
            
            if hasattr(response, "usage") and response.usage:
                parsed["llm_metadata"] = {
                    "model_name": response.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return parsed
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON output. Falling back.")
            return _fallback_answer(user_query, project_results)

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        # Fallback: generate a simple structured answer without LLM
        return _fallback_answer(user_query, project_results)


def _fallback_answer(user_query: str, results: list[ProjectResult]) -> dict:
    """Simple structured answer if LLM call fails — includes rich detail."""
    projects = []
    for proj in results:
        reasoning_lines = []
        
        address = proj.extra_props.get("address")
        location_str = address if address else f"{proj.neighbourhood}, {proj.city}"
        reasoning_lines.append(f"- Location: {location_str}")
        
        reasoning_lines.append(f"- Developer: {proj.developer}")
        
        # Project status
        status_val = proj.extra_props.get("project_status")
        if status_val and status_val.upper() not in ("UNKNOWN", "NONE"):
            reasoning_lines.append(f"- Status: {status_val.replace('_', ' ').title()}")
        
        # Possession date
        poss_date = proj.extra_props.get("possession_date")
        if poss_date:
            reasoning_lines.append(f"- Possession: {poss_date}")
        
        # RERA
        rera = proj.extra_props.get("rera_number")
        if rera:
            reasoning_lines.append(f"- RERA: {rera}")
        
        # Buildings
        bldgs = proj.extra_props.get("total_buildings")
        if bldgs:
            reasoning_lines.append(f"- Buildings: {bldgs}")
        
        if proj.units:
            unit_types = list({u.get("unit_type", "?") for u in proj.units})
            reasoning_lines.append(f"- Units: {', '.join(unit_types)}")
            
            # Show rooms from first unit
            for u in proj.units[:1]:
                rooms = u.get("rooms") or []
                room_strs = []
                for r in rooms[:6]:
                    rname = r.get("name", "")
                    rarea = r.get("area_sqft")
                    rlen = r.get("length")
                    rwid = r.get("width")
                    parts = []
                    if rlen and rwid:
                        parts.append(f"{rlen} x {rwid}")
                    if rarea:
                        try:
                            parts.append(f"{float(rarea):.1f} sqft")
                        except (ValueError, TypeError):
                            parts.append(f"{rarea} sqft")
                    if rname:
                        dim_str = f" ({', '.join(parts)})" if parts else ""
                        room_strs.append(f"{rname}{dim_str}")
                if room_strs:
                    reasoning_lines.append(f"- Rooms ({u.get('unit_type', '?')}): {'; '.join(room_strs)}")
        
        if getattr(proj, "floor_layouts", None):
            layout_details = []
            for f in proj.floor_layouts:
                name = f.get('layout_name', 'Unnamed')
                units = f.get('total_units_on_floor', '?')
                layout_details.append(f"{name} ({units} units/floor)")
            reasoning_lines.append(f"- Floor Layouts: {', '.join(layout_details)}")
        
        if proj.amenities:
            reasoning_lines.append(f"- Amenities: {', '.join(proj.amenities[:8])}")
        
        if proj.landmarks:
            reasoning_lines.append(f"- Nearby: {', '.join(proj.landmarks[:5])}")
            
        projects.append({
            "project_name": proj.project_name,
            "reasoning": "\n".join(reasoning_lines)
        })

    return {
        "general_summary": f"Here are the residential projects matching your query: **{user_query}**\n",
        "projects": projects,
        "conclusion": "Hope this helps!"
    }

