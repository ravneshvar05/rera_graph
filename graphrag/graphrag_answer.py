"""
graphrag_answer.py — LLM-powered answer generator.

Takes the assembled context from graphrag_retriever.py and the original user
query, and generates a natural language recommendation response.

The LLM acts as the "synthesis" layer of GraphRAG:
  graph context + vector context → structured LLM prompt → human answer
"""

from groq import Groq
from loguru import logger

from graphrag_config import settings
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
- Explain clearly WHY each recommended project fits the user's needs
- Mention key details: location, unit types, area (sqft), amenities, nearby landmarks
- If multiple projects match, list them in order of relevance
- If a project only partially matches, mention what it does and doesn't have
- If no projects match well, say so honestly and suggest what's available nearby
- Keep your tone friendly, professional, and concise
- Use bullet points for project details for easy reading
- Format numbers clearly: "850 sqft", "₹45–60 Lakhs", "2 BHK"

Important rules:
- ONLY use information from the provided context — do not invent data
- If price information is not in the context, do not guess prices
- If area is not in the context, say area information is not available
- Always ground your recommendations in the actual retrieved data
- End with a brief summary or next steps suggestion
"""


# ── Answer generator ──────────────────────────────────────────────────────────

def generate_answer(
    user_query: str,
    context_text: str,
    project_results: list[ProjectResult],
) -> str:
    """
    Generate a natural language recommendation answer using Gemini.

    Args:
        user_query: The original user query string.
        context_text: The assembled context from DualRetriever.
        project_results: Structured project results (for fallback display).

    Returns:
        A formatted string answer suitable for display in the chat UI.
    """
    if not project_results:
        return (
            "I couldn't find any projects matching your criteria in our current database. "
            "Try broadening your search — for example, remove a specific filter like the "
            "area range or amenity requirement."
        )

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
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        logger.success(f"Answer generated ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        # Fallback: generate a simple structured answer without LLM
        return _fallback_answer(user_query, project_results)


def _fallback_answer(user_query: str, results: list[ProjectResult]) -> str:
    """Simple structured answer if LLM call fails."""
    lines = [
        f"Here are the residential projects matching your query: **{user_query}**\n"
    ]
    for i, proj in enumerate(results, 1):
        lines.append(f"**{i}. {proj.project_name}**")
        lines.append(f"   - Location: {proj.neighbourhood}, {proj.city}")
        lines.append(f"   - Developer: {proj.developer}")
        if proj.units:
            unit_types = list({u.get("unit_type", "?") for u in proj.units})
            lines.append(f"   - Units: {', '.join(unit_types)}")
        if proj.amenities:
            lines.append(f"   - Amenities: {', '.join(proj.amenities[:5])}")
        lines.append("")
    return "\n".join(lines)
