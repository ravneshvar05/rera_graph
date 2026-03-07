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
from pydantic import BaseModel, Field
from loguru import logger

from graphrag_config import settings

# ── Configure Groq client ───────────────────────────────────────────────────
_groq_client = Groq(api_key=settings.GROQ_API_KEY)


# ── Intent Schema ─────────────────────────────────────────────────────────────

class QueryIntent(BaseModel):
    """Structured representation of what the user is looking for."""

    query_type: str = Field(
        "SPECIFIC",
        description="SPECIFIC (user wants particular units) or GLOBAL (user wants overview/all projects)",
    )
    bhk: Optional[Union[int, List[int]]] = Field(None, description="Number of BHK rooms requested (1, 2, 3, 4, 5) or list of numbers")
    property_type: Optional[Union[str, List[str]]] = Field(
        None, description="APARTMENT, VILLA, TENEMENT, BUNGALOW, or None or list of types"
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
    developer: Optional[str] = Field(None, description="Developer name if user specified one")
    semantic_keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Subjective/fuzzy keywords for vector search: spacious, luxury, family-friendly, affordable, etc.",
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """You are an expert real estate assistant for Indian residential projects (primarily Ahmedabad and Surat, Gujarat).

Your job is to extract structured search intent from a user's query about real estate.

Return ONLY a valid JSON object matching this schema — no explanation, no markdown, no code fences.

Schema:
{
  "query_type": "SPECIFIC" or "GLOBAL",
  "bhk": <integer, list of integers, or null>,
  "property_type": <"APARTMENT" | "VILLA" | "TENEMENT" | "BUNGALOW" | list of types | null>,
  "city": <string, list of strings, or null>,
  "neighbourhood": <string, list of strings, or null>,
  "zone": <string, list of strings, or null, e.g. "West Ahmedabad">,
  "amenities": [<list of amenity strings> or null],
  "landmark_types": [<from: EDUCATION, HEALTHCARE, COMMERCIAL, TRANSPORT, RELIGIOUS, RECREATION> or null],
  "specific_landmarks": [<list of specific place names> or null],
  "min_sqft": <number or null>,
  "max_sqft": <number or null>,
  "min_price_lakhs": <number or null>,
  "max_price_lakhs": <number or null>,
  "has_balcony": <true/false/null>,
  "has_parking": <true/false/null>,
  "entrance_facing": <"East"|"West"|"North"|"South"|null>,
  "developer": <string or null>,
  "semantic_keywords": [<subjective keywords like "spacious", "luxury", "affordable"> or null]
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
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_intent(user_query: str) -> QueryIntent:
    """
    Parse a natural language query into a structured QueryIntent.

    Falls back to a minimal intent (just semantic keywords) if the LLM fails
    so the system always produces some result.
    """
    try:
        response = _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model added them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        intent = QueryIntent(**data)
        logger.info(f"Intent parsed: bhk={intent.bhk}, location={intent.neighbourhood or intent.city}, "
                    f"amenities={intent.amenities}, type={intent.query_type}")
        return intent

    except Exception as e:
        logger.warning(f"Intent parsing failed ({e}), falling back to semantic-only mode.")
        return QueryIntent(
            query_type="SPECIFIC",
            semantic_keywords=user_query.split(),
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
