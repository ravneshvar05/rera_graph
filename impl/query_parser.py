"""
query_parser.py — Uses Groq LLM to extract structured filters from a user's
natural language query.

Input:  raw user message (string)
Output: ParsedQuery dataclass with typed fields
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq
from config import settings
from logger import logger
import re as _re


@dataclass
class ParsedQuery:
    """Structured representation of what the user is looking for."""
    # Hard filters (exact match in SQLite)
    city: Optional[str] = None
    neighbourhood: Optional[str] = None
    project_name: Optional[str] = None
    bhk: Optional[int] = None
    property_type: Optional[str] = None    # APARTMENT / VILLA / ROW_HOUSE / TENEMENT
    min_area_sqft: Optional[float] = None
    max_area_sqft: Optional[float] = None

    # Room feature flags
    must_have_pooja_room: bool = False
    must_have_study_room: bool = False
    must_have_terrace: bool = False
    must_have_servant_room: bool = False
    must_have_garden: bool = False
    must_have_home_theatre: bool = False
    must_have_gym: bool = False

    # Project-level
    must_have_pool: bool = False
    must_have_clubhouse: bool = False
    must_have_park: bool = False

    # Semantic search string (used for ChromaDB)
    semantic_query: str = ""

    # Whether we have enough info to search, or need to ask more
    needs_clarification: bool = False
    clarification_question: str = ""


_SYSTEM_PROMPT = """\
You are a real estate assistant helping users find the right home.
Your job is to extract structured search filters from the user's message.

IMPORTANT RULES:
- If the user provides a meaningful descriptive query (e.g., "affordable luxurious apartment"), ALWAYS set needs_clarification = false.
- ONLY ask for clarification (needs_clarification = true) if the user's input is extremely short or lacks any descriptive keywords, AND they haven't specified city, BHK, or property type.
- If the user IS asking about a specific project by name, or provides a rich semantic description, DO NOT require BHK, property type, or city, and set needs_clarification = false.
- If you have enough info to search (either structured filters OR a strong semantic_query), set needs_clarification = false.

PROPERTY TYPE VOCABULARY (CRITICAL):
  ROW_HOUSE  ← raw house, rawhouse, row house, row-house,
                bungalow, bunglow, banglow, bangalow, kothi, independent house
  VILLA      ← villa, villas, luxury villa, duplex villa, twin villa, farmhouse
  APARTMENT  ← apartment, flat, flats, studio, studio apartment
  TENEMENT   ← tenement, tenaments, tenements, chawl, chawls, gala, galas, old style flat
  PENTHOUSE  ← penthouse, penthouses, sky villa, sky apartment, rooftop apartment

CRITICAL: Generic words like "house", "houses", "home", "homes", "property",
"place", "residential" are NOT specific types. They mean ALL types.
ALWAYS set property_type = null for these generic words.
Examples:
  "2 bhk house in sarkhej" → property_type = null
  "3 bhk home near bopal"  → property_type = null
  "bungalow in bopal"      → property_type = "ROW_HOUSE"
  "2 bhk flat in sarkhej"  → property_type = "APARTMENT"

- property_type must be one of: APARTMENT, VILLA, ROW_HOUSE, TENEMENT, PENTHOUSE, null
- city: extract city name if mentioned, else null
- neighbourhood: extract neighbourhood or specific area if mentioned (e.g., 'Vinzol', 'Bopal'), else null
- project_name: extract specific project name if mentioned, else null
- bhk: integer only (1, 2, 3, 4, 5), null if not mentioned
- semantic_query: a clean English sentence combining all user preferences for semantic search
- All boolean flags default to false unless user explicitly mentions them
- Return ONLY valid JSON, no extra text.

JSON schema to return:
{
  "city": string | null,
  "neighbourhood": string | null,
  "project_name": string | null,
  "bhk": integer | null,
  "property_type": string | null,
  "min_area_sqft": float | null,
  "max_area_sqft": float | null,
  "must_have_pooja_room": boolean,
  "must_have_study_room": boolean,
  "must_have_terrace": boolean,
  "must_have_servant_room": boolean,
  "must_have_garden": boolean,
  "must_have_home_theatre": boolean,
  "must_have_gym": boolean,
  "must_have_pool": boolean,
  "must_have_clubhouse": boolean,
  "must_have_park": boolean,
  "semantic_query": string,
  "needs_clarification": boolean,
  "clarification_question": string
}
"""


# ─── Property-type word lists for post-processing guard ──────────────────────
_SPECIFIC_TYPE_WORDS = {
    "apartment", "apartments", "flat", "flats", "studio", "studio apartment",
    "villa", "villas", "farmhouse", "duplex villa", "twin villa",
    "row house", "rowhouse", "raw house", "rawhouse",
    "bungalow", "bunglow", "banglow", "bangalow", "kothi",
    "independent house",
    "tenement", "tenements", "tenaments", "chawl", "chawls", "gala", "galas",
    "penthouse", "penthouses", "sky villa", "sky apartment",
}

def _has_specific_type_word(text: str) -> bool:
    lower = text.lower()
    for w in _SPECIFIC_TYPE_WORDS:
        if " " in w:
            if w in lower:
                return True
        else:
            if _re.search(rf"\b{_re.escape(w)}\b", lower):
                return True
    return False


def parse_query(user_message: str, conversation_history: list[dict] | None = None) -> ParsedQuery:
    """
    Send user message to Groq and extract structured search filters.

    Args:
        user_message: The latest user input.
        conversation_history: Optional list of previous {role, content} dicts
                              so the LLM has context.

    Returns:
        ParsedQuery with all extracted filters.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Add conversation history for context (last 6 turns max)
    if conversation_history:
        messages.extend(conversation_history[-6:])

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.1,        # low temperature = more deterministic JSON
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        # POST-PROCESSING GUARD: Override property_type if user only used generic words
        if parsed.get("property_type") and not _has_specific_type_word(user_message):
            logger.info(
                f"Guard: parser returned property_type='{parsed['property_type']}' but user "
                f"used no specific type words. Overriding to null."
            )
            parsed["property_type"] = None

        result = ParsedQuery(**{k: v for k, v in parsed.items() if k in ParsedQuery.__dataclass_fields__})
        logger.debug(f"Parsed query: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Groq returned invalid JSON: {e}. Raw: {raw!r}")
        # Fallback — ask for clarification
        return ParsedQuery(
            needs_clarification=True,
            clarification_question="Could you tell me more about what you're looking for? For example: how many bedrooms (BHK), property type (apartment or villa), and which city?",
            semantic_query=user_message,
        )
    except Exception as e:
        logger.error(f"Query parsing failed: {e}")
        return ParsedQuery(
            needs_clarification=True,
            clarification_question="I couldn't process your request. Could you describe what you're looking for in simple terms?",
            semantic_query=user_message,
        )
