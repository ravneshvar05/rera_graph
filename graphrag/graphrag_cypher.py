"""
graphrag_cypher.py — Text-to-Cypher LLM module.

Given a user's natural language query, this module:
  1. Sends the query to an LLM with the full Neo4j schema + normalization tables
  2. Receives a structured JSON response containing a parameterized Cypher query,
     optional Cypher params, query_type (GLOBAL/SPECIFIC), and a vector_query string
  3. Returns these to graphrag_retriever.py for execution

This replaces the old hardcoded filter-builder in graphrag_retriever.py.
The LLM can handle ANY field in the schema — room sizes, unit areas,
entrance facing, project status, BHK, amenity tags, etc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import Groq
from loguru import logger

from graphrag_config import settings

# ── Groq client ────────────────────────────────────────────────────────────────
_groq = Groq(api_key=settings.GROQ_API_KEY)


# ── Output data class ──────────────────────────────────────────────────────────

@dataclass
class CypherQuery:
    """Result from the Text-to-Cypher LLM."""
    cypher:       str            # Parameterized Cypher WITH RETURN and LIMIT
    params:       dict           # Parameters for the Cypher query
    query_type:   str            # "GLOBAL" or "SPECIFIC"
    vector_query: str            # Query string for ChromaDB semantic search


# ── Schema context given to the LLM ───────────────────────────────────────────

_SCHEMA_CONTEXT = """
=== NEO4J SCHEMA ===

NODES:
- Project {project_id, project_name, developer_name, project_status (UNDER_CONSTRUCTION/READY_TO_MOVE/NEW_LAUNCH/UNKNOWN), possession_date, address, pin_code, has_clubhouse (0/1), has_pool (0/1), has_park (0/1), has_sports_courts (0/1), has_parking (0/1), has_commercial_shops (0/1), total_buildings, total_villas, society_description}
- City {name}
- Neighbourhood {name}
- Zone {name}
- Developer {name}
- Unit {unit_id, project_id, unit_type, property_type (APARTMENT/VILLA/ROW_HOUSE/TENEMENT/PENTHOUSE), bhk (integer), entrance_facing (East/West/North/South), carpet_sqft, super_builtup_sqft, balcony_sqft, wash_sqft, applicable_buildings, description}
- Room {room_type (BEDROOM/KITCHEN/DRAWING_ROOM/DINING/TOILET/BATHROOM/BALCONY/POOJA_ROOM/STORE_ROOM/STUDY_ROOM/SERVANT_ROOM/WASH_AREA/TERRACE/PASSAGE/LOBBY/DRESSING_ROOM/COURTYARD/WC/OTHER), name (canonical string), length, width, area_sqft, floor_level, attached_bathroom (0/1), has_balcony_access (0/1)}
- Amenity {name, category (SPORTS/WELLNESS/SECURITY/NATURE/SOCIAL/INFRASTRUCTURE), canonical_tags (list of strings, e.g. ["Gym", "Swimming Pool"])}
- Landmark {name, landmark_type (EDUCATION/HEALTHCARE/COMMERCIAL/TRANSPORT/RELIGIOUS/RECREATION)}
- FloorLayout {layout_id, layout_name, total_units_on_floor, has_lifts (0/1), has_staircases (0/1)}

RELATIONSHIPS:
(Project)-[:LOCATED_IN]->(Neighbourhood)
(Neighbourhood)-[:IN_CITY]->(City)
(Neighbourhood)-[:PART_OF]->(Zone)
(Project)-[:BUILT_BY]->(Developer)
(Project)-[:HAS_UNIT]->(Unit)
(Unit)-[:HAS_ROOM]->(Room)
(Project)-[:HAS_AMENITY]->(Amenity)
(Project)-[:NEAR]->(Landmark)
(Project)-[:HAS_FLOOR_LAYOUT]->(FloorLayout)

=== ROOM NAME NORMALIZATION ===
The database stores ONLY canonical room names. When writing WHERE clauses for rooms,
always use the canonical name (right column), regardless of what the user said:

User says                              → Use in Cypher (r.name =)
-----------------------------------------------------------------------
drawing room, living room, lounge,     → 'Hall'
  family room, l-d, drg room
master bedroom, master bed, mast bed   → 'Master Bedroom'
bedroom, bed room, bed                 → 'Bedroom'
dining room, dinning, dining           → 'Dining'
kitchen                                → 'Kitchen'
toilet, WC, w.c, powder room           → 'Toilet'
bathroom, bath room                    → 'Bathroom'
balcony, balc, deck, verandah          → 'Balcony'
wash area, wash, utility, laundry      → 'Wash Area'
pooja room, puja, puja room, mandir    → 'Pooja Room'
store room, store, storage room        → 'Store Room'
study room, study                      → 'Study Room'
servant room, maid room, maid          → 'Servant Room'
dressing room, dress room              → 'Dressing Room'
lobby, foyer, entrance                 → 'Lobby'
passage, corridor                      → 'Passage'
terrace                                → 'Terrace'

=== AMENITY CANONICAL TAGS ===
Amenity nodes have a canonical_tags list property for precise matching.
Use: EXISTS { MATCH (p)-[:HAS_AMENITY]->(am) WHERE $tag IN am.canonical_tags }

Common tags (use exactly as shown):
"Gym", "Swimming Pool", "Jogging Track", "Cricket", "Volleyball", "Basketball",
"Badminton", "Tennis", "Indoor Sports", "Outdoor Sports", "Spa", "Yoga",
"Meditation", "Health Club", "Clubhouse", "Party Area", "Amphitheatre",
"Children Play Area", "Senior Citizen Area", "Library", "Indoor Games",
"Mini Theatre", "Garden", "Lawn", "Park", "Courtyard", "Fountain", "Green Space",
"CCTV", "Security", "Gated Community", "Parking", "Lifts", "Power Backup",
"WiFi", "Water Supply", "Borewell", "Commercial Shops"
"""


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert for an Indian residential real estate knowledge graph.

{_SCHEMA_CONTEXT}

=== YOUR TASK ===
Given a user's natural language query about real estate, return a JSON object with:

{{
  "query_type": "SPECIFIC" or "GLOBAL",
  "cypher": "<complete parameterized Cypher query>",
  "params": {{ "<key>": <value>, ... }},
  "vector_query": "<clean text for semantic search>"
}}

=== CYPHER RULES ===

1. ALWAYS start with:
   MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
   OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
   OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
   OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
   OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
   WITH p, n, c, dev,
        collect(DISTINCT properties(u)) AS units,
        collect(DISTINCT am.name) AS amenities,
        collect(DISTINCT lm.name) AS landmarks

2. For ROOM filters, use EXISTS subqueries (NOT inline MATCH):
   EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r) WHERE r.name = $room_name }}
   EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r) WHERE r.name = $room_name AND r.area_sqft > $min_area }}

3. For AMENITY filters, use canonical_tags:
   EXISTS {{ MATCH (p)-[:HAS_AMENITY]->(am2) WHERE $amenity_tag IN am2.canonical_tags }}
   OR for well-known flags: p.has_pool = 1, p.has_clubhouse = 1, p.has_park = 1, p.has_parking = 1

4. For BHK filter: ANY(u IN units WHERE u.bhk = $bhk)

5. For AREA filters on units: ANY(u IN units WHERE u.carpet_sqft >= $min_sqft OR u.super_builtup_sqft >= $min_sqft)

6. For CITY filter: toLower(c.name) = toLower($city)

7. For NEIGHBOURHOOD filter: toLower(n.name) CONTAINS toLower($neighbourhood)

8. For PROJECT STATUS: toLower(p.project_status) CONTAINS toLower($status)
   Map user language: "ready to move"→"READY_TO_MOVE", "under construction"→"UNDER_CONSTRUCTION", "new launch"→"NEW_LAUNCH"

9. ALWAYS end with:
   RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
          units, amenities, landmarks
   LIMIT $limit
   And always include "limit": 50 in params.

10. query_type rules:
    - "GLOBAL": user wants all/overview with NO filters (e.g. "show all projects", "list everything")
    - "SPECIFIC": everything else — any filter at all

11. For GLOBAL queries, use this simple Cypher (no WHERE):
    MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
    OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
    OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
    OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
    WITH p, n, c, dev, collect(DISTINCT properties(u)) AS units, collect(DISTINCT am.name) AS amenities, collect(DISTINCT lm.name) AS landmarks
    RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer, units, amenities, landmarks
    LIMIT $limit

12. Use parameterized values (e.g. $bhk, $city) NOT inline literals in WHERE clauses.

13. vector_query: a clean English text summarizing the search intent for semantic vector search. Include location, BHK, property type, keywords. Do NOT include Cypher syntax.

Return ONLY the raw JSON object. No explanation, no markdown, no code fences.
"""


# ── Fallback Cypher builder ────────────────────────────────────────────────────

def _fallback_cypher(city: Optional[str] = None, bhk: Optional[int] = None) -> CypherQuery:
    """
    Minimal fallback Cypher when LLM generation fails.
    Filters by city and/or BHK if available, otherwise returns all.
    """
    match_base = """MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT properties(u)) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks"""

    wheres = []
    params: dict[str, Any] = {"limit": 50}

    if city:
        wheres.append("toLower(c.name) = toLower($city)")
        params["city"] = city
    if bhk is not None:
        wheres.append("ANY(u IN units WHERE u.bhk = $bhk)")
        params["bhk"] = bhk

    where_str = f"\nWHERE {' AND '.join(wheres)}" if wheres else ""
    cypher = (
        f"{match_base}{where_str}\n"
        "RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,\n"
        "       units, amenities, landmarks\n"
        "LIMIT $limit"
    )
    return CypherQuery(
        cypher=cypher,
        params=params,
        query_type="SPECIFIC" if wheres else "GLOBAL",
        vector_query=f"{bhk} BHK {city or ''}".strip(),
    )


# ── Main generation function ───────────────────────────────────────────────────

def generate_cypher(user_query: str) -> CypherQuery:
    """
    Generate a precise Cypher query from the user's natural language query.

    Returns a CypherQuery with:
    - cypher: parameterized Cypher ready for Neo4j execution
    - params: parameter dict for the Cypher
    - query_type: "GLOBAL" or "SPECIFIC"
    - vector_query: text string for ChromaDB semantic search

    Falls back to a simple city/BHK filter query if LLM generation fails.
    """
    try:
        response = _groq.chat.completions.create(
            model=settings.GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        cypher = data.get("cypher", "").strip()
        params = data.get("params", {})
        query_type = data.get("query_type", "SPECIFIC")
        vector_query = data.get("vector_query", user_query)

        # Safety: ensure limit param is present
        if "limit" not in params:
            params["limit"] = 50

        if not cypher:
            raise ValueError("LLM returned empty cypher")

        logger.info(
            f"[Text-to-Cypher] type={query_type} | params={list(params.keys())} | "
            f"vector_query={vector_query!r}"
        )
        logger.debug(f"[Text-to-Cypher] Cypher:\n{cypher}")

        return CypherQuery(
            cypher=cypher,
            params=params,
            query_type=query_type,
            vector_query=vector_query,
        )

    except Exception as e:
        logger.warning(f"[Text-to-Cypher] LLM generation failed ({e}). Using fallback.")
        return _fallback_cypher()


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "3 BHK flat in Ahmedabad with pooja room",
        "flat with hall bigger than 150 sqft",
        "east-facing villa with swimming pool",
        "ready to move 2 BHK under 1200 sqft carpet area",
        "show me all projects",
        "apartment with gym and parking near school in Bopal",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = generate_cypher(q)
        print(f"Type: {result.query_type}")
        print(f"Params: {result.params}")
        print(f"Vector: {result.vector_query}")
        print(f"Cypher:\n{result.cypher}")
