"""
graphrag_cypher.py — Text-to-Cypher LLM module (NLP-to-Cypher engine).

Given a user's natural language query, this module:
  1. Sends the query to an LLM with the full Neo4j schema + normalization tables
  2. Receives a structured JSON response containing a parameterized Cypher query,
     optional Cypher params, query_type, answer_columns, and vector_query
  3. Returns these to graphrag_retriever.py for execution

This module is designed to handle ANY type of buyer query — just like a
SQL assistant that converts NLP to SQL, this converts NLP to Cypher.
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
    cypher:         str        # Parameterized Cypher WITH RETURN and LIMIT
    params:         dict       # Parameters for the Cypher query
    query_type:     str        # "GLOBAL" / "SPECIFIC" / "LOOKUP" / "AGGREGATE"
    vector_query:   str        # Query string for ChromaDB semantic search
    answer_columns: list[str]  # Which RETURN columns hold the direct answer (e.g. ["answer_data"])


# ── Schema context given to the LLM ───────────────────────────────────────────

_SCHEMA_CONTEXT = """
=== NEO4J SCHEMA ===

NODES:
- Project {project_id, project_name, developer_name, project_status (UNDER_CONSTRUCTION/READY_TO_MOVE/NEW_LAUNCH/UNKNOWN), possession_date, address, pin_code, has_clubhouse (0/1), has_pool (0/1), has_park (0/1), has_sports_courts (0/1), has_parking (0/1), has_commercial_shops (0/1), total_buildings, total_villas, society_description, rera_number, building_names (list), road_widths (list), source_file}
- City {name}
- Neighbourhood {name}
- Zone {name}
- Developer {name}
- Unit {unit_id, project_id, unit_type, property_type (APARTMENT/VILLA/ROW_HOUSE/TENEMENT/PENTHOUSE), bhk (integer), entrance_facing (East/West/North/South), carpet_sqft, super_builtup_sqft, balcony_sqft, wash_sqft, applicable_buildings, description}
- Room {room_type (BEDROOM/KITCHEN/DRAWING_ROOM/DINING/TOILET/BATHROOM/BALCONY/POOJA_ROOM/STORE_ROOM/STUDY_ROOM/SERVANT_ROOM/WASH_AREA/TERRACE/PASSAGE/LOBBY/DRESSING_ROOM/COURTYARD/WC/OTHER), name (canonical string), length, width, area_sqft (stored as FLOAT), floor_level, attached_bathroom (0/1), has_balcony_access (0/1)}
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

=== CRITICAL DATA-TYPE RULES ===
Many numeric fields are stored as STRINGS or mixed types. You MUST use
type-safe comparisons to prevent errors or wrong results:

1. Room area_sqft → stored as FLOAT. Always use toFloat():
   toFloat(r.area_sqft) > toFloat($min_area)    -- NEVER  r.area_sqft > $min_area

2. Unit carpet_sqft, super_builtup_sqft, balcony_sqft, wash_sqft → may be FLOAT or NULL.
   Always use toFloat() for comparisons:
   toFloat(u.carpet_sqft) >= toFloat($min_sqft)

3. Unit bhk → stored as INTEGER. Safe to compare directly: u.bhk = $bhk
   But if filtering from collected list, use: toInteger(u.bhk) = toInteger($bhk)

4. Project total_buildings, total_villas → may be INT or NULL.
   Use toInteger() if comparing: toInteger(p.total_buildings) > 5

5. Boolean flags (has_clubhouse, has_pool, etc.) → stored as INTEGER (0/1).
   Compare as: p.has_pool = 1

6. For ALL string comparisons, ALWAYS use toLower() on both sides:
   toLower(p.project_name) CONTAINS toLower($project_name)

7. For NULL-safety, always add IS NOT NULL checks before numeric comparisons:
   r.area_sqft IS NOT NULL AND toFloat(r.area_sqft) > toFloat($min_area)

=== ROOM NAME NORMALIZATION ===
The database stores ONLY canonical room names. When writing WHERE clauses for rooms,
always use the canonical name (right column), regardless of what the user said:

User says                              → Use in Cypher (r.name =)
-----------------------------------------------------------------------
drawing room, living room, lounge,     → 'Hall'
  family room, l-d, drg room, hall,
  living, sitting room, main room
master bedroom, master bed, mast bed   → 'Master Bedroom'
bedroom, bed room, bed, room           → 'Bedroom'
dining room, dinning, dining, dinner   → 'Dining'
kitchen, kitchenette, cook             → 'Kitchen'
toilet, restroom, lavatory, loo        → 'Toilet'
bathroom, bath room, bath, washroom    → 'Bathroom'
balcony, balc, deck, verandah, patio   → 'Balcony'
wash area, wash, utility, laundry      → 'Wash Area'
pooja room, puja, puja room, mandir,   → 'Pooja Room'
  prayer room, worship room
store room, store, storage room,       → 'Store Room'
  storage, storeroom
study room, study, office room         → 'Study Room'
servant room, maid room, maid,         → 'Servant Room'
  helper room, staff room
dressing room, dress room, walk-in     → 'Dressing Room'
  closet, wardrobe room
lobby, foyer, entrance                 → 'Lobby'
passage, corridor, hallway             → 'Passage'
terrace, roof, rooftop                 → 'Terrace'
courtyard                              → 'Courtyard'
WC, w.c, water closet, powder room    → 'WC'

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

=== AMENITY KEYWORD → CANONICAL TAG MAPPING ===
User says "gym" / "fitness" → tag = "Gym"
User says "pool" / "swimming" → tag = "Swimming Pool"
User says "club" / "clubhouse" → tag = "Clubhouse"
User says "garden" / "park" / "green" → tag = "Garden" or "Park"
User says "security" / "cctv" / "gated" → tag = "Security" or "CCTV" or "Gated Community"
User says "parking" → tag = "Parking"
User says "sports" → tag = "Indoor Sports" or "Outdoor Sports"
User says "play area" / "kids" / "children" → tag = "Children Play Area"
User says "jogging" / "running" → tag = "Jogging Track"
"""


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert for an Indian residential real estate knowledge graph.
You function like an NLP-to-SQL engine, but for Cypher. Given ANY natural-language question
a property buyer might ask, you produce a precise, parameterized Cypher query that extracts
the answer from the graph database.

{_SCHEMA_CONTEXT}

=== YOUR TASK ===
Given a user's natural language query about real estate, return a JSON object with:

{{
  "query_type": "SPECIFIC" | "GLOBAL" | "LOOKUP" | "AGGREGATE",
  "cypher": "<complete parameterized Cypher query>",
  "params": {{ "<key>": <value>, ... }},
  "vector_query": "<clean text for semantic search>",
  "answer_columns": ["<column names in RETURN that hold the direct answer, e.g. answer_data>"]
}}

=== QUERY TYPE DECISION GUIDE ===

Choose the query type based on what the user is asking for:

── GLOBAL ──
When: user wants all/overview with NO filters ("show all projects", "list everything")
answer_columns: []

── SPECIFIC ──
When: user filters/searches by city, BHK, amenity, property type, facing, location, developer,
      area, status, landmark type, etc. — they are SEARCHING for matching projects.
answer_columns: []

── LOOKUP ──
When: user asks about a SPECIFIC NAMED project — details, room sizes, amenities, address,
      developer, RERA number, status, units, any attribute of a known project.
      Also use for: "does <project> have <amenity>?", "what is the address of <project>?"
answer_columns: ["answer_data"]

── AGGREGATE ──
When: user filters by a MEASURED VALUE on rooms (size comparisons), counts, min/max,
      or asks across projects for a numeric aggregation.
      Also use for: "find projects with hall > 100 sqft", "largest kitchen across all projects",
      "how many projects have a study room", "average hall size"
answer_columns: ["answer_data"]


=== CYPHER TEMPLATES ===

All templates use this common base for fetching projects with full context:

--- BASE MATCH (use for GLOBAL, SPECIFIC, LOOKUP, AGGREGATE) ---
MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }} ELSE null END) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks

--- RETURN clause (append to all queries) ---
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks
LIMIT $limit


── TEMPLATE 1: GLOBAL (all projects, no filters) ──
Just use BASE MATCH + RETURN. No WHERE clause.


── TEMPLATE 2: SPECIFIC (filtered search) ──
Use BASE MATCH, then add WHERE clause with filters. Common filters:

  BHK:           ANY(u IN units WHERE u.bhk = $bhk)
  City:          toLower(c.name) = toLower($city)
  Neighbourhood: toLower(n.name) CONTAINS toLower($neighbourhood)
  Status:        toLower(p.project_status) CONTAINS toLower($status)
  Room existence: EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r) WHERE r.name = $room_name }}
  Amenity tag:   EXISTS {{ MATCH (p)-[:HAS_AMENITY]->(am2) WHERE $tag IN am2.canonical_tags }}
  Amenity flags: p.has_pool = 1, p.has_clubhouse = 1, p.has_park = 1, p.has_parking = 1
  Unit area:     ANY(u IN units WHERE u.carpet_sqft IS NOT NULL AND toFloat(u.carpet_sqft) >= toFloat($min_sqft))
  Facing:        ANY(u IN units WHERE toLower(u.entrance_facing) = toLower($facing))
  Developer:     toLower(dev.name) CONTAINS toLower($developer)
  Property type: ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))
  Landmark type: EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE lm2.landmark_type = $landmark_type }}
  Specific landmark: EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE toLower(lm2.name) CONTAINS toLower($landmark_name) }}
  Total buildings: p.total_buildings IS NOT NULL AND toInteger(p.total_buildings) >= toInteger($min_buildings)

  Combine multiple filters with AND.


── TEMPLATE 3: LOOKUP (project-specific queries) ──

3a. ROOM-SPECIFIC LOOKUP (user asks about a specific room in a project):
    "what is the hall size in OUM Orbit?", "kitchen dimensions in Svasaar Pravesh"

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
WHERE toLower(p.project_name) CONTAINS toLower($project_name)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }} ELSE null END) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks
WITH p, n, c, dev, units, amenities, landmarks,
     [u IN units WHERE u IS NOT NULL | {{
       unit_type: u.unit_type,
       bhk: u.bhk,
       property_type: u.property_type,
       rooms: [r IN u.rooms WHERE toLower(r.name) = toLower($room_name) | {{
         room_name: r.name,
         length: r.length,
         width: r.width,
         area_sqft: r.area_sqft
       }}]
     }}] AS answer_data
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks, answer_data
LIMIT $limit

3b. FULL PROJECT DETAIL LOOKUP (user asks general details about a project):
    "tell me about OUM Orbit", "details of Svasaar Pravesh", "what is available in project X?"

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
WHERE toLower(p.project_name) CONTAINS toLower($project_name)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }} ELSE null END) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks
WITH p, n, c, dev, units, amenities, landmarks,
     [u IN units WHERE u IS NOT NULL | {{
       unit_type: u.unit_type,
       bhk: u.bhk,
       property_type: u.property_type,
       carpet_sqft: u.carpet_sqft,
       super_builtup_sqft: u.super_builtup_sqft,
       balcony_sqft: u.balcony_sqft,
       entrance_facing: u.entrance_facing,
       description: u.description,
       rooms: u.rooms
     }}] AS answer_data
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks, answer_data
LIMIT $limit

3c. PROJECT INFO LOOKUP (user asks a specific fact: address, RERA, developer, status, etc.):
    "what is the address of OUM Orbit?", "RERA number of Svasaar Pravesh?", "who built OUM Orbit?"

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
WHERE toLower(p.project_name) CONTAINS toLower($project_name)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }} ELSE null END) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks
WITH p, n, c, dev, units, amenities, landmarks,
     {{
       project_name: p.project_name,
       address: p.address,
       city: c.name,
       neighbourhood: n.name,
       developer: dev.name,
       rera_number: p.rera_number,
       project_status: p.project_status,
       possession_date: p.possession_date,
       total_buildings: p.total_buildings,
       total_villas: p.total_villas,
       society_description: p.society_description,
       has_clubhouse: p.has_clubhouse,
       has_pool: p.has_pool,
       has_park: p.has_park,
       has_parking: p.has_parking,
       has_sports_courts: p.has_sports_courts,
       building_names: p.building_names,
       pin_code: p.pin_code,
       amenities: amenities,
       landmarks: landmarks
     }} AS answer_data
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks, answer_data
LIMIT $limit

3d. AMENITY CHECK LOOKUP (user asks if a project has a specific amenity):
    "does OUM Orbit have swimming pool?", "amenities in Svasaar Pravesh?"

Use template 3b (full project detail) — the amenities list in the return will show whether
the project has the amenity. Set answer_columns = ["answer_data"].

3e. UNIT LISTING LOOKUP (user asks about unit types / configurations available):
    "what units are available in OUM Orbit?", "BHK options in Svasaar Pravesh"

Use template 3b (full project detail) — it includes all unit info.


── TEMPLATE 4: AGGREGATE (numeric comparisons, counts, statistics) ──

4a. ROOM SIZE COMPARISON:
    "find projects with hall > 100 sqft", "flats where kitchen < 50 sqft"

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
WITH p, n, c, dev,
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }} ELSE null END) AS units,
     collect(DISTINCT am.name) AS amenities,
     collect(DISTINCT lm.name) AS landmarks
WHERE EXISTS {{
  MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r2)
  WHERE r2.name = $room_name AND r2.area_sqft IS NOT NULL AND toFloat(r2.area_sqft) > toFloat($min_area)
}}
WITH p, n, c, dev, units, amenities, landmarks,
     [u IN units WHERE u IS NOT NULL | {{
       unit_type: u.unit_type,
       bhk: u.bhk,
       matching_rooms: [r IN u.rooms WHERE toLower(r.name) = toLower($room_name) AND r.area_sqft IS NOT NULL AND toFloat(r.area_sqft) > toFloat($min_area) | {{
         room_name: r.name, length: r.length, width: r.width, area_sqft: r.area_sqft
       }}]
     }}] AS answer_data
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks, answer_data
LIMIT $limit

Use > for "greater than / larger / bigger / more than", < for "less than / smaller / under".
Use toFloat() for ALL room area comparisons — NEVER compare raw strings.

4b. ROOM EXISTENCE FILTER (no size comparison):
    "projects with a study room", "flats that have pooja room"

Same as 4a but the WHERE clause is just:
WHERE EXISTS {{
  MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r2)
  WHERE r2.name = $room_name
}}

4c. COUNT / STATISTICS:
    "how many projects in Ahmedabad?", "how many 3 BHK projects?"

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
<optional WHERE filters>
WITH count(DISTINCT p) AS total_count,
     collect(DISTINCT p.project_name) AS project_names
RETURN {{ count: total_count, projects: project_names }} AS answer_data

4d. COMPARISON ACROSS PROJECTS:
    "which project has the largest hall?", "biggest kitchen in Ahmedabad"

MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
WHERE r.name = $room_name AND r.area_sqft IS NOT NULL
WITH p, r, toFloat(r.area_sqft) AS area
ORDER BY area DESC
LIMIT 10
MATCH (p)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
RETURN p.project_name AS project_name, r.name AS room_name,
       r.length AS length, r.width AS width, area AS area_sqft,
       n.name AS neighbourhood, c.name AS city, dev.name AS developer
LIMIT $limit


=== IMPORTANT RULES ===

1. PARAMETERIZE everything with $ — NEVER embed user values as inline literals.
2. Always include "limit": 50 in params (or a reasonable number). End every query with LIMIT $limit.
3. answer_columns MUST be ["answer_data"] for LOOKUP and AGGREGATE. For GLOBAL/SPECIFIC use [].
4. vector_query: clean English text summarizing search intent for semantic search.
5. Return ONLY raw JSON. No explanation, no markdown, no code fences.
6. For LOOKUP, extract ONLY the project name (strip city name, "project", "scheme", etc.):
   "OUM Orbit Ahmedabad" → $project_name = "OUM ORBIT"
   "Svasaar Pravesh project details" → $project_name = "Svasaar Pravesh"
7. Use CONTAINS (case-insensitive) for project name matching — NEVER use exact match (=).
8. When the user asks about a room, ALWAYS normalize to the canonical room name.
9. When city is not mentioned but a specific project or neighbourhood is mentioned,
   DO NOT add a city filter. The project name or neighbourhood filter is sufficient.
10. For area comparisons, always fence with IS NOT NULL: r.area_sqft IS NOT NULL AND toFloat(r.area_sqft) > ...
11. For string fields that might hold numbers, always cast: toFloat(), toInteger().
12. Handle both > and < operators. "at least 100" → >= 100. "under 100" → < 100. "between 100 and 200" → >= 100 AND <= 200.
13. If user asks about multiple rooms (e.g. "hall and kitchen sizes"), create answer_data with multiple room filters using OR.
14. For "does project X have Y amenity" questions, return amenity list and let the response indicate presence/absence.
15. If the user query is ambiguous or you're unsure, prefer RETURNING MORE DATA over returning nothing.
    It is much better to return extra data than to return zero results.
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
     collect(DISTINCT CASE WHEN u IS NOT NULL THEN u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] } ELSE null END) AS units,
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
        answer_columns=[],
    )


# ── Main generation function ───────────────────────────────────────────────────

def generate_cypher(user_query: str) -> CypherQuery:
    """
    Generate a precise Cypher query from the user's natural language query.

    Returns a CypherQuery with:
    - cypher: parameterized Cypher ready for Neo4j execution
    - params: parameter dict for the Cypher
    - query_type: "GLOBAL" / "SPECIFIC" / "LOOKUP" / "AGGREGATE"
    - vector_query: text string for ChromaDB semantic search
    - answer_columns: which RETURN columns hold the direct answer

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
            max_tokens=2048,
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
        answer_columns = data.get("answer_columns", [])

        # Safety: ensure limit param is present
        if "limit" not in params:
            params["limit"] = 50

        # Safety: LOOKUP/AGGREGATE must declare answer_columns
        if query_type in ("LOOKUP", "AGGREGATE") and not answer_columns:
            answer_columns = ["answer_data"]

        if not cypher:
            raise ValueError("LLM returned empty cypher")

        # ── Post-process: fix common LLM Cypher mistakes ──
        cypher = _post_process_cypher(cypher, params)

        logger.info(
            f"[Text-to-Cypher] type={query_type} | params={list(params.keys())} | "
            f"answer_cols={answer_columns} | vector_query={vector_query!r}"
        )
        logger.debug(f"[Text-to-Cypher] Cypher:\n{cypher}")

        return CypherQuery(
            cypher=cypher,
            params=params,
            query_type=query_type,
            vector_query=vector_query,
            answer_columns=answer_columns,
        )

    except Exception as e:
        logger.warning(f"[Text-to-Cypher] LLM generation failed ({e}). Using fallback.")
        return _fallback_cypher()


def _post_process_cypher(cypher: str, params: dict) -> str:
    """
    Fix common LLM-generated Cypher issues:
    - Ensure toFloat() wraps area_sqft comparisons
    - Ensure IS NOT NULL guards on numeric comparisons
    - Fix any raw string numeric comparisons
    """
    # Fix: r.area_sqft > $value → toFloat(r.area_sqft) > toFloat($value)
    # Match patterns like r.area_sqft > $min_area that aren't already wrapped
    cypher = re.sub(
        r'(?<!toFloat\()([a-zA-Z]\w*\.area_sqft)\s*([><=!]+)\s*(?!toFloat)',
        lambda m: f'toFloat({m.group(1)}) {m.group(2)} toFloat',
        cypher,
    )

    # Fix: u.carpet_sqft comparisons
    cypher = re.sub(
        r'(?<!toFloat\()([a-zA-Z]\w*\.carpet_sqft)\s*([><=!]+)\s*(?!toFloat)',
        lambda m: f'toFloat({m.group(1)}) {m.group(2)} toFloat',
        cypher,
    )

    # Fix: u.super_builtup_sqft comparisons
    cypher = re.sub(
        r'(?<!toFloat\()([a-zA-Z]\w*\.super_builtup_sqft)\s*([><=!]+)\s*(?!toFloat)',
        lambda m: f'toFloat({m.group(1)}) {m.group(2)} toFloat',
        cypher,
    )

    return cypher


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        # SPECIFIC
        "3 BHK flat in Ahmedabad with pooja room",
        "east-facing villa with swimming pool",
        "ready to move 2 BHK under 1200 sqft carpet area",
        "show me all projects",
        # LOOKUP — room-specific
        "what is the hall size in OUM Orbit Ahmedabad",
        "kitchen dimensions in Svasaar Pravesh",
        # LOOKUP — full project detail
        "find me the details of svasaar pravesh",
        "show me all rooms in OUM Orbit",
        "what units are available in OUM Orbit?",
        # LOOKUP — project info
        "what is the address of OUM Orbit?",
        "who is the developer of Svasaar Pravesh?",
        "RERA number of OUM Orbit",
        # LOOKUP — amenity check
        "does OUM Orbit have swimming pool?",
        "what amenities does Svasaar Pravesh have?",
        # AGGREGATE - size filter
        "find projects with hall size greater than 100 sqft",
        "find me flats where hall is less than 80 sqft",
        "show 2 BHK apartments with master bedroom bigger than 150 sqft",
        # AGGREGATE - existence filter
        "show me projects that have a study room",
        "find projects with a pooja room",
        "which projects have a servant room",
        # AGGREGATE — statistics
        "how many projects are in Ahmedabad?",
        "which project has the largest hall?",
        # SPECIFIC — developer
        "show all projects by Svasaar",
        # SPECIFIC — landmark
        "projects near hospital in Ahmedabad",
        # SPECIFIC — status
        "ready to move projects in Ahmedabad",
        # Without city
        "tell me about OUM Orbit",
        "2 BHK in Vinzol",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = generate_cypher(q)
        print(f"Type:   {result.query_type}")
        print(f"Params: {result.params}")
        print(f"AnsCol: {result.answer_columns}")
        print(f"Vector: {result.vector_query}")
        print(f"Cypher:\n{result.cypher}")
