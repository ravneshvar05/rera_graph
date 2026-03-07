"""
answer_generator.py — Uses Gemini Flash to generate a natural language answer
based on search results and the detected query type.

Query-type-aware:
  SEARCH    → Friendly scannable list with project highlights
  AGGREGATE → Summary statistics ("There are 47 3-BHK apartments...")
  COMPARE   → Side-by-side comparison of 2 named projects
  DETAIL    → Rich deep-dive: all units, rooms, full amenities
"""

import json
from google import genai
from google.genai import types as gtypes

from config import settings
from logger import logger


_BASE_SYSTEM = """\
You are a warm, knowledgeable real estate advisor helping Indian home buyers find their ideal property.
You speak naturally and helpfully, like an expert agent who knows every project deeply.

CORE RULES:
- NEVER make up data. Only use what is given.
- Do NOT use markdown headers (##, ###) in your response.
- Use emoji sparingly and only where it adds clarity.
- Always end with an invitation to ask more.
"""

_PROMPTS = {
    "SEARCH": _BASE_SYSTEM + """
You are presenting search results. Format rules:
- Brief warm opening (1 line max)
- For each project: 1-2 crisp lines covering: project name, developer, city/neighbourhood, 
  and the most relevant features for THIS user's query
- If a project has multiple matching unit variants, mention the options briefly
- Close with: "Would you like full details on any of these?"
""",

    "AGGREGATE": _BASE_SYSTEM + """
You are answering a statistics/counting question. Format rules:
- Give a direct, clear answer with the numbers up front
- Then provide a breakdown (by city, by BHK, by type) if given
- Keep it factual and scannable
- Close with: "Want me to filter these further or find specific projects?"
""",

    "COMPARE": _BASE_SYSTEM + """
You are comparing two or more real estate projects side-by-side. Format rules:
- Start with a 1-line summary of the key difference
- Cover: Location, BHK options, size range, key amenities, special features
- Clearly recommend which is better for which type of buyer
- Close with: "Would you like detailed room dimensions for either project?"
""",

    "DETAIL": _BASE_SYSTEM + """
You are providing a complete deep-dive on a specific project. Format rules:
- Project overview: developer, location, society highlights
- For each unit variant: BHK, size, entrance facing, special rooms
- Mention room dimensions if available (bedrooms, drawing room)
- List all amenities and nearby landmarks
- Close with: "Is there anything specific you'd like to know more about?"
""",
}


# ─── Result Formatters ────────────────────────────────────────────────────────

def _format_search_context(results: list[dict]) -> str:
    lines = []
    for i, p in enumerate(results, 1):
        lines.append(f"\n[Project {i}] {p.get('project_name')} by {p.get('developer_name')}")
        lines.append(f"  Location: {p.get('city')}, {p.get('neighbourhood') or ''}")
        if p.get("project_status") and p["project_status"] != "UNKNOWN":
            lines.append(f"  Status: {p['project_status']}")
        if p.get("possession_date"):
            lines.append(f"  Possession: {p['possession_date']}")

        # Amenities
        amenities = p.get("amenities", [])
        if amenities:
            lines.append(f"  Amenities: {', '.join(amenities[:8])}")

        # Landmarks (top 5)
        landmarks = p.get("nearby_landmarks", [])
        if landmarks:
            lines.append(f"  Nearby: {', '.join(landmarks[:5])}")

        # Unit variants
        units = p.get("matching_units", [])
        lines.append(f"  Unit Variants ({len(units)}):")
        for u in units[:6]:  # max 6 variants shown
            uline = f"    • {u.get('bhk')} BHK {u.get('property_type')}"
            if u.get("super_builtup_sqft"):
                uline += f" | {u['super_builtup_sqft']} sqft"
            elif u.get("carpet_area_sqft"):
                uline += f" | carpet {u['carpet_area_sqft']} sqft"
            if u.get("entrance_facing"):
                uline += f" | {u['entrance_facing']}-facing"
            # Special features
            feats = []
            for col, label in [
                ("has_pooja_room","Pooja"), ("has_study_room","Study"),
                ("has_terrace","Terrace"), ("has_garden","Garden"),
                ("has_home_theatre","Home Theatre"), ("has_gym","Gym"),
                ("has_dressing_room","Dressing Room"), ("has_courtyard","Courtyard"),
                ("has_servant_room","Servant Rm"),
            ]:
                if u.get(col):
                    feats.append(label)
            if feats:
                uline += f" | {', '.join(feats)}"
            lines.append(uline)

    return "\n".join(lines)


def _format_aggregate_context(agg: dict) -> str:
    lines = [f"Total matching units: {agg['total_units']}"]

    if agg.get("by_city"):
        lines.append("\nBreakdown by city:")
        for row in agg["by_city"]:
            lines.append(f"  {row['city'] or 'Unknown'}: {row['cnt']} project(s)")

    if agg.get("by_bhk"):
        lines.append("\nBreakdown by BHK:")
        for row in agg["by_bhk"]:
            lines.append(f"  {row['bhk']} BHK: {row['cnt']} unit(s)")

    if agg.get("by_property_type"):
        lines.append("\nBreakdown by property type:")
        for row in agg["by_property_type"]:
            lines.append(f"  {row['property_type']}: {row['cnt']} unit(s)")

    return "\n".join(lines)


def _format_detail_context(results: list[dict]) -> str:
    lines = []
    for p in results:
        lines.append(f"PROJECT: {p.get('project_name')} by {p.get('developer_name')}")
        lines.append(f"Location: {p.get('address') or ''}, {p.get('city')}, {p.get('neighbourhood') or ''}")
        if p.get("project_status") != "UNKNOWN":
            lines.append(f"Status: {p.get('project_status')}")
        if p.get("possession_date"):
            lines.append(f"Possession: {p.get('possession_date')}")
        if p.get("rera_number"):
            lines.append(f"RERA: {p.get('rera_number')}")

        lines.append(f"\nSociety: {p.get('society_description') or 'N/A'}")

        amenities = p.get("amenities", [])
        if amenities:
            lines.append(f"Amenities: {', '.join(amenities)}")

        landmarks = p.get("nearby_landmarks", [])
        if landmarks:
            lines.append(f"Nearby landmarks: {', '.join(landmarks[:12])}")

        lines.append("\n--- Unit Variants ---")
        for u in p.get("matching_units", []):
            lines.append(
                f"\n  {u.get('unit_type')} | {u.get('bhk')} BHK {u.get('property_type')}"
                f" | {u.get('super_builtup_sqft') or u.get('carpet_area_sqft') or 'N/A'} sqft"
                f" | Facing: {u.get('entrance_facing') or 'N/A'}"
            )
            if u.get("description"):
                lines.append(f"  Description: {u['description']}")

            # Rooms
            rooms = u.get("rooms", [])
            if rooms:
                key_rooms = [
                    r for r in rooms
                    if r.get("room_type") in (
                        "BEDROOM", "DRAWING_ROOM", "KITCHEN", "POOJA_ROOM",
                        "STUDY_ROOM", "TERRACE", "BALCONY", "COURTYARD",
                    ) or any(kw in (r.get("name") or "").lower()
                             for kw in ("theatre", "gym", "garden", "dressing"))
                ]
                for r in key_rooms[:12]:
                    dim = ""
                    if r.get("length") and r.get("width"):
                        dim = f" ({r['length']} × {r['width']})"
                    elif r.get("area_sqft"):
                        dim = f" ({r['area_sqft']} sqft)"
                    level = f" [{r['floor_level']}]" if r.get("floor_level") else ""
                    lines.append(f"    - {r.get('name')}{dim}{level}")

        lines.append("")
    return "\n".join(lines)


# ─── Main Generator ──────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    search_result: dict,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate a natural language answer using Groq (LLaMA 70B).

    Args:
        user_query: Original user message.
        search_result: Output of search.search() → {query_type, results, aggregate}
        conversation_history: Previous {role, content} turns.

    Returns:
        A natural language string answer.
    """
    from groq import Groq

    if not settings.GROQ_API_KEY:
        return "Groq API key not set. Cannot generate response."

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return "Failed to initialize LLM client."

    qt = search_result.get("query_type", "SEARCH").upper()
    results = search_result.get("results", [])
    agg = search_result.get("aggregate")
    location_tier = search_result.get("location_tier")            # "exact"|"landmark_proximity"|None
    searched_locations = search_result.get("searched_locations", [])  # ["Nikol","Iscon Ambli Road"]

    # ── Build context ──
    if qt == "AGGREGATE" and agg:
        context = _format_aggregate_context(agg)
    elif qt == "DETAIL" and results:
        context = _format_detail_context(results)
    elif results:
        context = _format_search_context(results)
    else:
        context = "No matching projects found in the database."

    # ── No results fallback ──
    if not results and not agg:
        loc_phrase = ""
        if searched_locations:
            loc_phrase = " in " + " or ".join(searched_locations)
        return (
            f"I searched through all our projects but couldn't find any properties{loc_phrase} matching your criteria. "
            "Try broadening your search — for example, removing one filter like BHK count or area size. "
            "Or ask me to show all projects in a specific city."
        )

    # ── Location proximity note (landmark-only hits) ──
    # When results come from landmark matches (weak tier), NOT from projects actually
    # located in that area, we MUST tell the LLM to disclose this to the user.
    location_note = ""
    if location_tier == "landmark_proximity" and searched_locations:
        loc_display = " / ".join(searched_locations)
        location_note = (
            f"\n\nIMPORTANT — LOCATION TRANSPARENCY: The user asked for properties IN '{loc_display}', "
            f"but NO projects were found with '{loc_display}' in their address or neighbourhood. "
            f"The results below are projects that have '{loc_display}' mentioned only as a NEARBY LANDMARK "
            f"(e.g., near Iscon Temple, near Nikol bus stand). "
            f"You MUST start your response with: 'I couldn't find any projects located directly in {loc_display}. "
            f"However, here are some projects near {loc_display} landmarks:' "
            f"Do NOT present these as being 'in {loc_display}'."
        )
    fallback_non_apt = search_result.get("fallback_non_apt", False)
    original_type = search_result.get("original_type") or ""
    fallback_note = ""
    if fallback_non_apt and original_type:
        type_display = original_type.replace("_", " ").title()  # e.g. "Row House"
        fallback_note = (
            f"\n\nIMPORTANT CONTEXT: The user asked for a '{type_display}' specifically, "
            f"but NO {type_display} listings were found in the database for their location/filters. "
            f"The results below are the BEST AVAILABLE non-apartment alternatives "
            f"(Villas, Row Houses, Tenements, Penthouses) that match their other criteria. "
            f"You MUST clearly tell the user upfront: 'I couldn't find any {type_display}s matching your "
            f"criteria, but here are the best non-apartment options available:'"
        )

    # ── Generic search note (user said "house"/"home" → all types searched) ──
    if not original_type and results and qt == "SEARCH":
        # Check if results contain multiple property types → inform user
        all_types = set()
        for p in results:
            for u in p.get("matching_units", []):
                pt = u.get("property_type", "")
                if pt:
                    all_types.add(pt.replace("_", " ").title())
        if len(all_types) > 1:
            types_str = ", ".join(sorted(all_types))
            fallback_note += (
                f"\n\nCONTEXT: The user used a generic term (like 'house' or 'home') "
                f"so results include ALL property types: {types_str}. "
                f"Briefly mention that you're showing options across different property types."
            )

    # Conversation context (last 4 turns)
    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]
        history_text = "\nPrevious conversation:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        ) + "\n"

    system = _PROMPTS.get(qt, _PROMPTS["SEARCH"])
    system_prompt = system + f"\n\nData retrieved from database:\n{context}{location_note}{fallback_note}"

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if history_text:
        messages.append({"role": "user", "content": history_text + f"\nUser's request: {user_query}"})
    else:
        messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"Generated {qt} answer ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return (
            "I found some results but had trouble generating a response right now. "
            "Please try again in a moment."
        )
