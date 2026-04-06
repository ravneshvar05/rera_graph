"""
graphrag_intent.py — LLM-powered query intent parser.

Converts a user's natural language query into a structured QueryIntent object
that drives downstream Neo4j Cypher queries and ChromaDB vector search.

Example:
    intent = parse_intent("2 BHK flat in Vinzol near school with swimming pool")
    # QueryIntent(bhk=2, location="Vinzol", amenities=["swimming pool"],
    #             landmark_types=["EDUCATION"], query_type="SPECIFIC")
"""

import json
import re
from typing import Optional, Union, List
from groq import Groq
import google.generativeai as genai
from pydantic import BaseModel, Field
from loguru import logger

from graphrag_config import settings, build_gemini_pool, build_groq_pool


# ── Intent Schema ─────────────────────────────────────────────────────────────

class QueryIntent(BaseModel):
    """Structured representation of what the user is looking for."""

    query_type: str = Field(
        "SPECIFIC",
        description="SPECIFIC (user wants particular units) or GLOBAL (user wants overview/all projects)",
    )
    bhk: Optional[Union[int, List[int]]] = Field(None, description="Number of BHK rooms requested (1, 2, 3, 4, 5) or list of numbers")
    property_type: Optional[Union[str, List[str]]] = Field(
        None, description="APARTMENT, VILLA, TENEMENT, BUNGALOW, PENTHOUSE, ROW_HOUSE, or None or list of types"
    )
    city: Optional[Union[str, List[str]]] = Field(None, description="City name or list of cities, e.g. Ahmedabad, Surat")
    neighbourhood: Optional[Union[str, List[str]]] = Field(
        None, description="Locality/neighbourhood or list of localities, e.g. Vinzol, Bopal, Nikol"
    )
    zone: Optional[Union[str, List[str]]] = Field(
        None,
        description="Zone or list of zones, e.g. 'West Ahmedabad', 'East Ahmedabad', 'South Ahmedabad'",
    )
    amenities: Optional[List[str]] = Field(
        default_factory=list,
        description="List of amenity names user wants, e.g. ['swimming pool', 'gym', 'clubhouse']",
    )
    landmark_types: Optional[List[str]] = Field(
        default_factory=list,
        description="Types of landmarks user wants nearby: EDUCATION, HEALTHCARE, COMMERCIAL, TRANSPORT, RELIGIOUS, RECREATION",
    )
    specific_landmarks: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific landmark names user mentioned, e.g. ['Karnavati Club', 'Airport']",
    )
    min_sqft: Optional[float] = Field(None, description="Minimum carpet/super-builtup area in sqft")
    max_sqft: Optional[float] = Field(None, description="Maximum carpet/super-builtup area in sqft")
    min_price_lakhs: Optional[float] = Field(None, description="Minimum price in lakhs")
    max_price_lakhs: Optional[float] = Field(None, description="Maximum price in lakhs")
    has_balcony: Optional[bool] = Field(None, description="User wants balcony")
    has_parking: Optional[bool] = Field(None, description="User wants parking")
    entrance_facing: Optional[str] = Field(None, description="Facing direction: East, West, North, South")
    developer: Optional[Union[str, List[str]]] = Field(None, description="Developer name(s) if user specified one or more")
    project_names: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific project names the user mentioned (e.g., ['Oum Orbit', 'Svasar Pravesh'])",
    )
    semantic_keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Subjective/fuzzy keywords for vector search: spacious, luxury, family-friendly, affordable, etc.",
    )
    min_units_per_floor: Optional[int] = Field(
        None, description="Minimum number of units per floor requested"
    )
    max_units_per_floor: Optional[int] = Field(
        None, description="Maximum number of units per floor requested"
    )
    area_qualifier: Optional[str] = Field(
        None,
        description="'carpet' if user explicitly says 'carpet area/sqft'; 'super_builtup' if user says 'super built-up area/SBA/built-up area'. null if just 'area' or 'sqft' with no qualifier."
    )
    around_area: Optional[bool] = Field(
        None,
        description="True if user says 'around', 'approximately', 'roughly', 'about', '~', 'close to', 'approx'. Null/False otherwise."
    )
    room_dimensions: Optional[List[dict]] = Field(
        default_factory=list,
        description=(
            "List of room dimension constraints the user specified. "
            "Each entry: {\"room\": \"Bedroom\", \"d1\": 10.6, \"d2\": 12.4, \"dim_qualifier\": null}. "
            "d1/d2 are the two dimensions in feet (feet.inches notation: 12'6\" \u2192 12.6). "
            "dim_qualifier: null/\"exact\" = exact match (default, no tolerance); "
            "\"around\" = ±15%% on each dimension independently (NOT area multiplication); "
            "\"gte\" = both dimensions >= given values (at least / bigger than / minimum); "
            "\"lte\" = both dimensions <= given values (less than / smaller than / under). "
            "Only populate when user gives explicit NxM / N by M / N x M dimensions for a named room."
        )
    )
    num_floors: Optional[int] = Field(
        None,
        description=(
            "Number of floors the unit spans, extracted when user explicitly mentions "
            "floor count (e.g. '3 floor house', '2 storey', 'duplex', 'single floor'). "
            "Ground-only = 1, Ground+First = 2, Ground+First+Second = 3, etc. "
            "Leave null if user says 'ground floor' without specifying total count, "
            "or when no floor count information is present."
        )
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """You are an expert real estate assistant for Indian residential projects (primarily Ahmedabad and Surat, Gujarat).

Your job is to extract structured search intent from a user's query about real estate.

Return ONLY a valid JSON object matching this schema — no explanation, no markdown, no code fences.

Schema:
{
  "query_type": "SPECIFIC" or "GLOBAL",
  "bhk": <integer, list of integers, or null>,
  "property_type": <"APARTMENT" | "VILLA" | "TENEMENT" | "BUNGALOW" | "PENTHOUSE" | "ROW_HOUSE" | list of types | null>,
  "city": <string, list of strings, or null>,
  "neighbourhood": <string, list of strings, or null>,
  "zone": <string, list of strings, or null, e.g. "West Ahmedabad">,
  "amenities": [<list of amenity strings> or null],
  "landmark_types": [<from: EDUCATION, HEALTHCARE, COMMERCIAL, TRANSPORT, RELIGIOUS, RECREATION> or null],
  "specific_landmarks": [<list of specific place names> or null],
  "min_sqft": <number or null>,
  "max_sqft": <number or null>,
  "area_qualifier": "carpet" | "super_builtup" | null,
  "around_area": true | false | null,
  "min_price_lakhs": <number or null>,
  "max_price_lakhs": <number or null>,
  "has_balcony": <true/false/null>,
  "has_parking": <true/false/null>,
  "entrance_facing": <"East"|"West"|"North"|"South"|null>,
  "developer": <string, list of strings, or null>,
  "project_names": [<list of specific project names> or null],
  "min_units_per_floor": <integer or null>,
  "max_units_per_floor": <integer or null>,
  "num_floors": <integer or null>,
  "semantic_keywords": [<subjective keywords like "spacious", "luxury", "affordable"> or null],
  "room_dimensions": [
    {"room": "<canonical room name>", "d1": <float feet>, "d2": <float feet>},
    ...
  ]
}

Rules:
- Use query_type="GLOBAL" if user asks for "all projects", "list everything", "show all", "what's available", or wants a broad overview.
- Use query_type="SPECIFIC" for any filtered or preference-based search.
- For area/zone: if user says "west Ahmedabad" → zone="West Ahmedabad". If says "Bopal" → neighbourhood="Bopal".
- For BHK: "2 bedroom", "two BHK", "2BHK" all → bhk=2.
- If user mentions "school nearby" → landmark_types=["EDUCATION"]. "near hospital" → landmark_types=["HEALTHCARE"].
- Extract semantic/fuzzy words into semantic_keywords: "nice", "spacious", "luxury", "affordable", "family-friendly" etc.
- If a field is not mentioned, use null or empty list.
- Locality names in Gujarat: Vinzol, Bopal, Nikol, Naroda, Vatva, Gamdi, Satellite, Chandkheda, Thaltej, Vastrapur, etc.
- area_qualifier: Set "carpet" if user says "carpet area", "carpet sqft", "by carpet". Set "super_builtup" if user says "super built-up area", "super builtup", "built-up area", "SBA". Leave null if user just says "area" or "sqft" with no such qualifier.
- around_area: Set true if user says "around", "approximately", "roughly", "about", "~", "close to", "near about", "approx". Leave null/false for "at least", "minimum", "more than", "less than", "exactly", or a plain numeric mention with no qualifier word.
- room_dimensions: Extract ONLY when user gives explicit dimensions (NxM, N by M, N x M, length N width M) for a named room.
  - Map the room name to its canonical name (bedroom→"Bedroom", kitchen→"Kitchen", etc.).
  - d1 and d2 are the two numeric values in feet, using feet.inches notation (12'6" → 12.6, 10'0" → 10.0).
  - Example: "bedroom 10.6 by 12.4" → [{"room": "Bedroom", "d1": 10.6, "d2": 12.4}]
  - Example: "bedroom 10x10, toilet 4x6" → [{"room": "Bedroom", "d1": 10.0, "d2": 10.0}, {"room": "Toilet", "d1": 4.0, "d2": 6.0}]
  - Leave empty list [] if no room dimensions mentioned.

=== PROPERTY TYPE RULES (READ CAREFULLY) ===

MAP these explicit terms to property_type:
  "flat" / "flats" / "apartment" / "apartments" → "APARTMENT"
  "villa" / "villas" / "villa type"              → "VILLA"
  "bungalow" / "bungalows" / "banglow"          → "BUNGALOW"
  "tenement" / "tenaments" / "tenement type"    → "TENEMENT"
  "pandhout" / "pandhu" / "pandhut"             → "TENEMENT"  (Gujarati term for tenement)
  "row house" / "rowhouse" / "row-house"        → "ROW_HOUSE"
  "penthouse" / "pent house" / "sky villa"      → "PENTHOUSE"

GENERIC terms — DO NOT set property_type (leave null):
  "house" / "houses" / "ghar" / "makan" / "home" / "homes" /
  "property" / "properties" / "residence" / "unit" / "units"
  These mean any residential property — never convert them to a property_type filter.

INDEPENDENT HOUSING context — set property_type as LIST ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"]:
  "independent house" / "independent property" / "standalone house" →
    property_type = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"]
  "house with ground floor" / "ground floor house" / "house on ground floor" →
    property_type = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"], num_floors = null
    REASON: apartments sit on ANY floor of a building; when a user says "house with
    ground floor" they mean an independent property whose ground floor is part of
    the unit — all such types have a ground floor regardless of total stories.
  "ground floor flat" / "ground floor apartment" →
    property_type = "APARTMENT" (explicit apartment context; ground floor = building floor 0)
  "ground floor 2 BHK" / "ground floor unit" (no explicit flat/house) →
    property_type = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"] (independent context assumed)

FLOOR COUNT context — use num_floors + property_type list:
  "N floor house" / "N storey house" / "N मजला घर" →
    num_floors = N, property_type = ["TENEMENT","VILLA","BUNGALOW"] (multi-floor non-apt)
  "duplex" →
    num_floors = 2, property_type = ["TENEMENT","VILLA","BUNGALOW"]
  "single floor house" / "one floor house" / "ground floor only house" →
    num_floors = 1, property_type = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"]
  "house with first floor" / "house up to first floor" →
    num_floors = 2, property_type = ["TENEMENT","VILLA","BUNGALOW"]
  "house with second floor" →
    num_floors = 3, property_type = ["TENEMENT","VILLA","BUNGALOW"]
  "N floor flat" / "N floor apartment" →
    property_type = "APARTMENT", num_floors = null
    (Apartments are single-floor units; "N floor" here means building floor level, not unit floors.
     Do NOT set num_floors for apartments.)
  "top floor apartment" / "top floor flat" →
    property_type = "APARTMENT", num_floors = null
  "N floor tenement" / "N floor villa" / "N floor bungalow" →
    num_floors = N, property_type = <explicit type>

num_floors should be null unless the user EXPLICITLY states a floor count.
"House with ground floor" does NOT imply num_floors=1 — leave num_floors null.
"House" alone does NOT imply any num_floors.
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_raw_intent(raw: str) -> QueryIntent:
    """Parse raw JSON string into a QueryIntent."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    intent = QueryIntent(**data)
    logger.info(
        f"Intent parsed: bhk={intent.bhk}, "
        f"location={intent.neighbourhood or intent.city}, "
        f"amenities={intent.amenities}, type={intent.query_type}"
    )
    return intent


def parse_intent(user_query: str, api_keys: dict = None) -> QueryIntent:
    """
    Parse a natural language query into a structured QueryIntent.

    Tries Groq pool first, then Gemini pool as fallback.
    Falls back to minimal keyword-based intent if all LLMs fail.
    """
    groq_pool   = build_groq_pool(api_keys)
    gemini_pool = build_gemini_pool(api_keys)

    # ── Primary: Groq pool ────────────────────────────────────────────────
    for attempt in range(len(groq_pool)):
        key = groq_pool.next()
        if not key:
            break
        try:
            groq_client = Groq(api_key=key)
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,
            )
            return _parse_raw_intent(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"[IntentParser] Groq slot {groq_pool.last_slot} failed ({e}).")

    # ── Fallback: Gemini pool ─────────────────────────────────────────────
    for attempt in range(len(gemini_pool)):
        key = gemini_pool.next()
        if not key:
            break
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=_INTENT_SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            response = model.generate_content(user_query)
            return _parse_raw_intent(response.text)
        except Exception as e:
            logger.warning(f"[IntentParser] Gemini slot {gemini_pool.last_slot} failed ({e}).")

    # ── Last-resort: simple keyword extraction ────────────────────────────
    logger.warning("[IntentParser] All LLM keys exhausted. Using keyword fallback.")
    lower_query = user_query.lower()

    # Detect known Gujarat localities so location context is never lost
    _KNOWN_LOCALITIES = [
        "vinzol", "sarkhej", "bopal", "nikol", "naroda", "vatva", "gamdi",
        "satellite", "chandkheda", "thaltej", "vastrapur", "hanspura", "paldi",
        "isanpur", "ghodasar", "kotarpur", "chiloda", "kubernagar", "naranpura",
        "vastral", "shantigram", "gokuldham",
    ]
    found_localities = [loc.title() for loc in _KNOWN_LOCALITIES if loc in lower_query]
    neighbourhood = found_localities if found_localities else None

    fallback_city = None
    common_cities = ["ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar"]
    for c in common_cities:
        if c in lower_query:
            fallback_city = c.title()
            break

    return QueryIntent(
        query_type="SPECIFIC",
        neighbourhood=neighbourhood if neighbourhood else None,
        city=fallback_city,
        semantic_keywords=[] if (neighbourhood or fallback_city) else user_query.split(),
    )


if __name__ == "__main__":
    # Quick test
    test_queries = [
        "2 BHK flat in Vinzol with swimming pool",
        "show me all available projects",
        "luxury villa near airport in West Ahmedabad",
        "affordable 1 BHK near school for family",
        "3 BHK apartment with gym, clubhouse and parking under 80 lakhs",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        intent = parse_intent(q)
        print(intent.model_dump_json(indent=2))
