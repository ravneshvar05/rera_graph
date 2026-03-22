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

import google.generativeai as genai
from groq import Groq
from loguru import logger

from graphrag_config import settings
from graphrag_intent import QueryIntent


# ── Generalized room synonym system ───────────────────────────────────────────
# Maps EVERY possible user keyword (lowercase) to the list of canonical room
# names that should be searched.  This is the single source of truth for room
# name expansion.  If a canonical name maps to itself only (e.g. "Hall"), a
# user query for "hall" will still work fine — no expansion needed.
#
# The key design principle: if a user could reasonably mean MULTIPLE canonical
# rooms with one word, ALL of them must be listed.
_USER_TERM_TO_ROOMS: dict[str, list[str]] = {
    # ── Wet-room cluster (washroom / restroom / toilet / bathroom / WC) ──
    "toilet":        ["Toilet", "Bathroom", "WC"],
    "restroom":      ["Toilet", "Bathroom", "WC"],
    "lavatory":      ["Toilet", "Bathroom", "WC"],
    "loo":           ["Toilet", "Bathroom", "WC"],
    "bathroom":      ["Toilet", "Bathroom", "WC"],
    "bath room":     ["Toilet", "Bathroom", "WC"],
    "bath":          ["Toilet", "Bathroom", "WC"],
    "washroom":      ["Toilet", "Bathroom", "WC"],
    "wash room":     ["Toilet", "Bathroom", "WC"],
    "wc":            ["Toilet", "Bathroom", "WC"],
    "w.c":           ["Toilet", "Bathroom", "WC"],
    "water closet":  ["Toilet", "Bathroom", "WC"],
    "powder room":   ["Toilet", "Bathroom", "WC"],
    # canonical names themselves
    "Toilet":        ["Toilet", "Bathroom", "WC"],
    "Bathroom":      ["Toilet", "Bathroom", "WC"],
    "WC":            ["Toilet", "Bathroom", "WC"],

    # ── Bedroom cluster ──
    "bedroom":       ["Bedroom", "Master Bedroom"],
    "bed room":      ["Bedroom", "Master Bedroom"],
    "bed":           ["Bedroom", "Master Bedroom"],
    "room":          ["Bedroom", "Master Bedroom"],
    "master bedroom":["Master Bedroom", "Bedroom"],
    "master bed":    ["Master Bedroom", "Bedroom"],
    "mast bed":      ["Master Bedroom", "Bedroom"],
    "Bedroom":       ["Bedroom", "Master Bedroom"],
    "Master Bedroom":["Master Bedroom", "Bedroom"],

    # ── Living / Hall ──
    "hall":          ["Hall"],
    "drawing room":  ["Hall"],
    "living room":   ["Hall"],
    "lounge":        ["Hall"],
    "family room":   ["Hall"],
    "l-d":           ["Hall"],
    "drg room":      ["Hall"],
    "living":        ["Hall"],
    "sitting room":  ["Hall"],
    "main room":     ["Hall"],
    "Hall":          ["Hall"],

    # ── Dining ──
    "dining room":   ["Dining"],
    "dinning":       ["Dining"],
    "dining":        ["Dining"],
    "dinner":        ["Dining"],
    "Dining":        ["Dining"],

    # ── Kitchen ──
    "kitchen":       ["Kitchen"],
    "kitchenette":   ["Kitchen"],
    "cook":          ["Kitchen"],
    "Kitchen":       ["Kitchen"],

    # ── Balcony / Terrace (users may confuse these) ──
    "balcony":       ["Balcony", "Terrace"],
    "balc":          ["Balcony", "Terrace"],
    "deck":          ["Balcony", "Terrace"],
    "verandah":      ["Balcony", "Terrace"],
    "patio":         ["Balcony", "Terrace"],
    "terrace":       ["Terrace", "Balcony"],
    "roof":          ["Terrace"],
    "rooftop":       ["Terrace"],
    "Balcony":       ["Balcony", "Terrace"],
    "Terrace":       ["Terrace", "Balcony"],

    # ── Utility / Wash ──
    "wash area":     ["Wash Area"],
    "utility":       ["Wash Area"],
    "laundry":       ["Wash Area"],
    "Wash Area":     ["Wash Area"],

    # ── Lobby / Passage / Entrance (users may confuse these) ──
    "lobby":         ["Lobby", "Passage"],
    "foyer":         ["Lobby", "Passage"],
    "entrance":      ["Lobby", "Passage"],
    "passage":       ["Passage", "Lobby"],
    "corridor":      ["Passage", "Lobby"],
    "hallway":       ["Passage", "Lobby"],
    "Lobby":         ["Lobby", "Passage"],
    "Passage":       ["Passage", "Lobby"],

    # ── Prayer / Pooja ──
    "pooja room":    ["Pooja Room"],
    "puja":          ["Pooja Room"],
    "puja room":     ["Pooja Room"],
    "mandir":        ["Pooja Room"],
    "prayer room":   ["Pooja Room"],
    "worship room":  ["Pooja Room"],
    "Pooja Room":    ["Pooja Room"],

    # ── Storage ──
    "store room":    ["Store Room"],
    "store":         ["Store Room"],
    "storage room":  ["Store Room"],
    "storage":       ["Store Room"],
    "storeroom":     ["Store Room"],
    "Store Room":    ["Store Room"],

    # ── Study ──
    "study room":    ["Study Room"],
    "study":         ["Study Room"],
    "office room":   ["Study Room"],
    "Study Room":    ["Study Room"],

    # ── Servant / Maid ──
    "servant room":  ["Servant Room"],
    "maid room":     ["Servant Room"],
    "maid":          ["Servant Room"],
    "helper room":   ["Servant Room"],
    "staff room":    ["Servant Room"],
    "Servant Room":  ["Servant Room"],

    # ── Dressing ──
    "dressing room": ["Dressing Room"],
    "dress room":    ["Dressing Room"],
    "walk-in closet":["Dressing Room"],
    "wardrobe room": ["Dressing Room"],
    "Dressing Room": ["Dressing Room"],

    # ── Courtyard ──
    "courtyard":     ["Courtyard"],
    "Courtyard":     ["Courtyard"],
}


def _expand_room_name(room_name: str) -> list[str]:
    """Given any room name (canonical or user term), return the full list of
    canonical names to search.  Falls back to [room_name] if unrecognized."""
    result = _USER_TERM_TO_ROOMS.get(room_name)
    if result:
        return result
    result = _USER_TERM_TO_ROOMS.get(room_name.lower())
    if result:
        return result
    # Substring fallback: e.g. "attached bathroom" contains "bathroom"
    lower = room_name.lower()
    for keyword, canonical_list in _USER_TERM_TO_ROOMS.items():
        if keyword in lower or lower in keyword:
            return canonical_list
    return [room_name]  # unknown room — keep as-is


# ── Generalized amenity synonym system ─────────────────────────────────────────
# Maps EVERY possible user keyword (lowercase) to the list of canonical amenity
# tags that should be checked.  Covers all common ways users refer to amenities.
_USER_TERM_TO_AMENITY_TAGS: dict[str, list[str]] = {
    # ── Green / Outdoor spaces ──
    "park":           ["Garden", "Lawn", "Park", "Green Space"],
    "garden":         ["Garden", "Lawn", "Park", "Green Space"],
    "lawn":           ["Garden", "Lawn", "Park", "Green Space"],
    "green":          ["Garden", "Lawn", "Park", "Green Space"],
    "green space":    ["Garden", "Lawn", "Park", "Green Space"],
    "greenery":       ["Garden", "Lawn", "Park", "Green Space"],
    "landscaping":    ["Garden", "Lawn", "Park", "Green Space"],
    "courtyard":      ["Courtyard", "Garden"],
    "fountain":       ["Fountain", "Garden"],

    # ── Fitness / Health ──
    "gym":            ["Gym", "Health Club"],
    "gymnasium":      ["Gym", "Health Club"],
    "fitness":        ["Gym", "Health Club"],
    "fitness center": ["Gym", "Health Club"],
    "fitness centre": ["Gym", "Health Club"],
    "health club":    ["Gym", "Health Club"],
    "exercise":       ["Gym", "Health Club"],
    "workout":        ["Gym", "Health Club"],

    # ── Swimming ──
    "pool":           ["Swimming Pool"],
    "swimming":       ["Swimming Pool"],
    "swimming pool":  ["Swimming Pool"],

    # ── Clubhouse / Social ──
    "club":           ["Clubhouse", "Party Area"],
    "clubhouse":      ["Clubhouse"],
    "club house":     ["Clubhouse"],
    "community hall": ["Clubhouse", "Party Area"],
    "banquet":        ["Clubhouse", "Party Area"],
    "party":          ["Party Area", "Clubhouse"],
    "party hall":     ["Party Area", "Clubhouse"],
    "party area":     ["Party Area", "Clubhouse"],
    "amphitheatre":   ["Amphitheatre"],
    "amphitheater":   ["Amphitheatre"],

    # ── Security ──
    "security":       ["Security", "CCTV", "Gated Community"],
    "cctv":           ["Security", "CCTV", "Gated Community"],
    "gated":          ["Security", "CCTV", "Gated Community"],
    "gated community":["Gated Community", "Security", "CCTV"],
    "surveillance":   ["Security", "CCTV"],
    "guard":          ["Security", "Gated Community"],

    # ── Parking ──
    "parking":        ["Parking"],
    "car park":       ["Parking"],
    "garage":         ["Parking"],

    # ── Sports ──
    "sports":         ["Indoor Sports", "Outdoor Sports", "Cricket", "Badminton", "Tennis", "Basketball", "Volleyball"],
    "indoor sports":  ["Indoor Sports", "Indoor Games"],
    "outdoor sports": ["Outdoor Sports", "Cricket", "Tennis"],
    "cricket":        ["Cricket", "Outdoor Sports"],
    "badminton":      ["Badminton", "Indoor Sports"],
    "tennis":         ["Tennis", "Outdoor Sports"],
    "basketball":     ["Basketball", "Outdoor Sports"],
    "volleyball":     ["Volleyball", "Outdoor Sports"],
    "indoor games":   ["Indoor Games", "Indoor Sports"],
    "games":          ["Indoor Games", "Indoor Sports", "Outdoor Sports"],

    # ── Children / Senior ──
    "play area":      ["Children Play Area"],
    "playground":     ["Children Play Area"],
    "children":       ["Children Play Area"],
    "kids":           ["Children Play Area"],
    "kids area":      ["Children Play Area"],
    "senior":         ["Senior Citizen Area"],
    "senior citizen": ["Senior Citizen Area"],
    "elderly":        ["Senior Citizen Area"],

    # ── Tracks / Paths ──
    "jogging":        ["Jogging Track"],
    "jogging track":  ["Jogging Track"],
    "running":        ["Jogging Track"],
    "running track":  ["Jogging Track"],
    "walking":        ["Jogging Track"],
    "walking track":  ["Jogging Track"],

    # ── Wellness / Relaxation ──
    "yoga":           ["Yoga", "Meditation", "Spa"],
    "meditation":     ["Meditation", "Yoga"],
    "spa":            ["Spa", "Yoga", "Meditation"],
    "wellness":       ["Spa", "Yoga", "Meditation", "Health Club"],
    "sauna":          ["Spa"],
    "steam":          ["Spa"],

    # ── Entertainment / Education ──
    "library":        ["Library"],
    "reading":        ["Library"],
    "theatre":        ["Mini Theatre"],
    "theater":        ["Mini Theatre"],
    "mini theatre":   ["Mini Theatre"],
    "movie":          ["Mini Theatre"],
    "cinema":         ["Mini Theatre"],

    # ── Infrastructure ──
    "power backup":   ["Power Backup"],
    "generator":      ["Power Backup"],
    "backup":         ["Power Backup"],
    "lift":           ["Lifts"],
    "lifts":          ["Lifts"],
    "elevator":       ["Lifts"],
    "wifi":           ["WiFi"],
    "internet":       ["WiFi"],
    "water supply":   ["Water Supply", "Borewell"],
    "borewell":       ["Borewell", "Water Supply"],
    "water":          ["Water Supply", "Borewell"],
    "commercial":     ["Commercial Shops"],
    "shop":           ["Commercial Shops"],
    "shops":          ["Commercial Shops"],
}


def _expand_amenity_tags(tag: str) -> list[str]:
    """Given any amenity keyword or canonical tag, return the full list of
    canonical tags to search.  Falls back to [tag] if unrecognized."""
    result = _USER_TERM_TO_AMENITY_TAGS.get(tag)
    if result:
        return result
    result = _USER_TERM_TO_AMENITY_TAGS.get(tag.lower())
    if result:
        return result
    # Substring fallback
    lower = tag.lower()
    for keyword, tag_list in _USER_TERM_TO_AMENITY_TAGS.items():
        if keyword in lower or lower in keyword:
            return tag_list
    return [tag]  # unknown amenity — keep as-is


# ── Output data class ──────────────────────────────────────────────────────────

@dataclass
class CypherQuery:
    """Result from the Text-to-Cypher LLM."""
    cypher:         str        # Parameterized Cypher WITH RETURN and LIMIT
    params:         dict       # Parameters for the Cypher query
    query_type:     str        # "GLOBAL" / "SPECIFIC" / "LOOKUP" / "AGGREGATE"
    vector_query:   str        # Query string for ChromaDB semantic search
    answer_columns: list[str]  # Which RETURN columns hold the direct answer (e.g. ["answer_data"])
    intent:         QueryIntent = field(default_factory=QueryIntent)  # Parsed intent (merged into single LLM call)
    tokens_used:    int = 0    # Total tokens used for this LLM call
    engine_used:    str = "Unknown"  # Which engine generated this (Gemini/Groq)


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
- Room {room_type, name (canonical string), length, width, area_sqft (FLOAT), floor_level, attached_bathroom (0/1), has_balcony_access (0/1)}
- Amenity {name, category (SPORTS/WELLNESS/SECURITY/NATURE/SOCIAL/INFRASTRUCTURE), canonical_tags (list of strings)}
- Landmark {name, landmark_type (EDUCATION/HEALTHCARE/COMMERCIAL/TRANSPORT/RELIGIOUS/RECREATION)}
- FloorLayout {layout_id, layout_name, total_units_on_floor, has_lifts (0/1), has_staircases (0/1)}

RELATIONSHIPS:
(Project)-[:LOCATED_IN]->(Neighbourhood)-[:IN_CITY]->(City)
(Neighbourhood)-[:PART_OF]->(Zone)
(Project)-[:BUILT_BY]->(Developer)
(Project)-[:HAS_UNIT]->(Unit)-[:HAS_ROOM]->(Room)
(Project)-[:HAS_AMENITY]->(Amenity)
(Project)-[:NEAR]->(Landmark)
(Project)-[:HAS_FLOOR_LAYOUT]->(FloorLayout)

=== DATA-TYPE RULES ===
- area_sqft → FLOAT. Always: toFloat(r.area_sqft) > toFloat($val). Fence with IS NOT NULL.
- carpet_sqft, super_builtup_sqft → FLOAT/NULL. Use toFloat() + IS NOT NULL check.
- bhk → INTEGER. Direct compare: u.bhk = $bhk
- Boolean flags (has_clubhouse etc.) → INTEGER 0/1: p.has_pool = 1
- String comparisons → toLower() both sides: toLower(p.project_name) CONTAINS toLower($name)
- NEVER apply toLower/toFloat/toInteger to a list. Use comprehension: [x IN $list | toLower(x)]

=== ROOM NAMES (canonical only) ===
Map user terms → canonical name. System auto-expands synonyms in post-processing.
  living room / drawing room / lounge / hall / sitting room / family room / l-d → "Hall"
  bedroom / bed room → "Bedroom"    master bedroom → "Master Bedroom"
  kitchen → "Kitchen"    dining / dining room → "Dining"
  toilet / bathroom / washroom / restroom / wc / powder room → any of "Toilet","Bathroom","WC"
  balcony / deck / verandah / patio → "Balcony"    terrace / rooftop → "Terrace"
  wash area / utility / laundry → "Wash Area"    pooja room / prayer room / mandir → "Pooja Room"
  store room / storage → "Store Room"    study room / office room → "Study Room"
  servant room / maid room → "Servant Room"    dressing room / walk-in closet → "Dressing Room"
  lobby / foyer → "Lobby"    passage / corridor / hallway → "Passage"    courtyard → "Courtyard"
Use r.name = $room_name — system auto-converts to IN [...] for ambiguous terms.

=== AMENITY TAGS ===
Match via canonical_tags list:
  EXISTS { MATCH (p)-[:HAS_AMENITY]->(am) WHERE ANY(t IN $amenity_tags WHERE t IN am.canonical_tags) }
All tags: "Gym","Health Club","Swimming Pool","Jogging Track","Cricket","Volleyball","Basketball",
"Badminton","Tennis","Indoor Sports","Outdoor Sports","Spa","Yoga","Meditation","Clubhouse",
"Party Area","Amphitheatre","Children Play Area","Senior Citizen Area","Library","Indoor Games",
"Mini Theatre","Garden","Lawn","Park","Courtyard","Fountain","Green Space","CCTV","Security",
"Gated Community","Parking","Lifts","Power Backup","WiFi","Water Supply","Borewell","Commercial Shops"
Key mappings: gym/fitness→["Gym","Health Club"], pool→["Swimming Pool"],
garden/park/green→["Garden","Park","Lawn","Green Space"],
security/cctv/gated→["Security","CCTV","Gated Community"],
kids/play area→["Children Play Area"], jogging→["Jogging Track"],
yoga/meditation→["Yoga","Meditation"], clubhouse→["Clubhouse"], parking→["Parking"]
System auto-expands tags in post-processing.
"""


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert for an Indian residential real estate knowledge graph.
Given ANY natural-language question a property buyer might ask, produce a precise, parameterized Cypher query.

{_SCHEMA_CONTEXT}

=== YOUR TASK ===
Return a JSON object:

{{
  "query_type": "SPECIFIC" | "GLOBAL" | "LOOKUP" | "AGGREGATE",
  "cypher": "<parameterized Cypher>",
  "params": {{ "<key>": <value>, ... }},
  "vector_query": "<clean text for semantic search>",
  "answer_columns": ["<RETURN columns with direct answer>"],
  "intent": {{
    "bhk": <int, list, or null>,
    "property_type": <"APARTMENT"|"VILLA"|"TENEMENT"|"BUNGALOW"|list|null>,
    "city": <string, list, or null>,
    "neighbourhood": <string, list, or null>,
    "zone": <string, list, or null>,
    "amenities": [<EXTRACT STRICTLY: Do NOT guess amenities from vague words like 'luxurious' or 'peaceful'. Only extract if user explicitly names them (e.g., 'pool', 'gym').>],
    "landmark_types": [<EDUCATION|HEALTHCARE|COMMERCIAL|TRANSPORT|RELIGIOUS|RECREATION>],
    "specific_landmarks": [<place names>],
    "min_sqft": <number or null>, "max_sqft": <number or null>,
    "min_price_lakhs": <number or null>, "max_price_lakhs": <number or null>,
    "has_balcony": <bool or null>, "has_parking": <bool or null>,
    "entrance_facing": <"East"|"West"|"North"|"South"|null>,
    "developer": <string, list, or null>,
    "project_names": [<project names>],
    "semantic_keywords": [<PUT VAGUE/SUBJECTIVE ADJECTIVES HERE: e.g., 'peaceful', 'luxurious', 'affordable', 'premium'>],
    "min_units_per_floor": <int or null>, "max_units_per_floor": <int or null>
  }}
}}

=== QUERY TYPES ===
GLOBAL: no filters ("show all projects"). answer_columns: []
SPECIFIC: filtered search (city, BHK, amenity, etc.). answer_columns: []
LOOKUP: about a NAMED project (details, rooms, address, RERA, etc.). answer_columns: ["answer_data"]
AGGREGATE: measured values, counts, comparisons across projects. answer_columns: ["answer_data"]


=== CYPHER TEMPLATES ===

All templates use this common base for fetching projects with full context:

--- BASE MATCH (use for GLOBAL, SPECIFIC, LOOKUP, AGGREGATE) ---
MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
  WITH u WHERE u IS NOT NULL
  RETURN collect(u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }}) AS units
}}
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
  WITH am WHERE am IS NOT NULL
  RETURN collect(am.name) AS amenities
}}
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
  WITH lm WHERE lm IS NOT NULL
  RETURN collect(lm.name) AS landmarks
}}
WITH p, n, c, dev, units, amenities, landmarks

--- RETURN clause (append to all queries) ---
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks
LIMIT $limit


── TEMPLATE 1: GLOBAL (all projects, no filters) ──
Just use BASE MATCH + RETURN. No WHERE clause.


── TEMPLATE 2: SPECIFIC (filtered search) ──
Use BASE MATCH entirely (including the WITH clause), then add a WHERE clause AFTER the WITH clause to filter.
Example:
... BASE MATCH WITH ...
WHERE ANY(u IN units WHERE u.bhk = $bhk) AND toLower(c.name) = toLower($city)
... RETURN ...

Common filters:

  BHK:           ANY(u IN units WHERE u.bhk = $bhk)
  City:          toLower(c.name) = toLower($city)
  Neighbourhood: (toLower(n.name) CONTAINS toLower($neighbourhood) OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower($neighbourhood) OR toLower(p.address) CONTAINS toLower($neighbourhood))
  Multi-neighbourhood: ANY(x IN $neighbourhoods WHERE toLower(n.name) CONTAINS toLower(x) OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower(x) OR toLower(p.address) CONTAINS toLower(x))
  Status:        toLower(p.project_status) CONTAINS toLower($status)
  Room existence: EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r) WHERE r.name = $room_name }}
  Amenity tag:   EXISTS {{ MATCH (p)-[:HAS_AMENITY]->(am2) WHERE $tag IN am2.canonical_tags }}
  Amenity flags: p.has_pool = 1, p.has_clubhouse = 1, p.has_park = 1, p.has_parking = 1
  Unit area:     ANY(u IN units WHERE u.carpet_sqft IS NOT NULL AND toFloat(u.carpet_sqft) >= toFloat($min_sqft))
  Facing:        ANY(u IN units WHERE toLower(u.entrance_facing) = toLower($facing))
  Developer:     toLower(dev.name) CONTAINS toLower($developer)
  Property type: ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))
  Landmark type: EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE lm2.landmark_type = $landmark_type }}
  Specific landmark: (EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE toLower(lm2.name) CONTAINS toLower($landmark_name) }} OR toLower(p.address) CONTAINS toLower($landmark_name))
  Total buildings: p.total_buildings IS NOT NULL AND toInteger(p.total_buildings) >= toInteger($min_buildings)

  Combine multiple filters with AND.


── TEMPLATE 3: LOOKUP (project-specific queries) ──
All LOOKUP queries use BASE + WHERE toLower(p.project_name) CONTAINS toLower($project_name),
then add a second WITH to compute answer_data. Include answer_data in RETURN.

3a. ROOM LOOKUP ("hall size in OUM Orbit?"): answer_data =
  [u IN units WHERE u IS NOT NULL | {{
    unit_type: u.unit_type, bhk: u.bhk, property_type: u.property_type,
    rooms: [r IN u.rooms WHERE toLower(r.name) = toLower($room_name) |
      {{ room_name: r.name, length: r.length, width: r.width, area_sqft: r.area_sqft }}]
  }}]

3b. FULL DETAIL ("tell me about OUM Orbit", amenity check, unit listing): answer_data =
  [u IN units WHERE u IS NOT NULL | {{
    unit_type: u.unit_type, bhk: u.bhk, property_type: u.property_type,
    carpet_sqft: u.carpet_sqft, super_builtup_sqft: u.super_builtup_sqft,
    balcony_sqft: u.balcony_sqft, entrance_facing: u.entrance_facing,
    description: u.description, rooms: u.rooms
  }}]

3c. PROJECT INFO (\"address of OUM Orbit?\", \"RERA number?\"): answer_data =
  {{
    project_name: p.project_name, address: p.address, city: c.name, neighbourhood: n.name,
    developer: dev.name, rera_number: p.rera_number, project_status: p.project_status,
    possession_date: p.possession_date, total_buildings: p.total_buildings,
    total_villas: p.total_villas, society_description: p.society_description,
    has_clubhouse: p.has_clubhouse, has_pool: p.has_pool, has_park: p.has_park,
    has_parking: p.has_parking, has_sports_courts: p.has_sports_courts,
    building_names: p.building_names, pin_code: p.pin_code,
    amenities: amenities, landmarks: landmarks
  }}


── TEMPLATE 4: AGGREGATE (numeric comparisons, counts, statistics) ──

4a. ROOM SIZE COMPARISON ("hall > 100 sqft"):
Use the BASE MATCH pattern, then add a WHERE EXISTS filter, then return BOTH the project context
AND the matching room details in answer_data. Use this EXACT structure:

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
  WITH u WHERE u IS NOT NULL
  RETURN collect(u {{ .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }}) AS units
}}
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
  WITH am WHERE am IS NOT NULL
  RETURN collect(am.name) AS amenities
}}
CALL {{
  WITH p
  OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
  WITH lm WHERE lm IS NOT NULL
  RETURN collect(lm.name) AS landmarks
}}
WITH p, n, c, dev, units, amenities, landmarks
WHERE EXISTS {{
  MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
  WHERE r2.name = $room_name AND r2.area_sqft IS NOT NULL AND toFloat(r2.area_sqft) > toFloat($min_area)
}}
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks,
       {{ project_name: p.project_name, city: c.name, neighbourhood: n.name,
          matching_rooms: [u_item IN units WHERE u_item IS NOT NULL |
            {{ unit_type: u_item.unit_type, bhk: u_item.bhk,
               rooms: [r_item IN u_item.rooms WHERE r_item.name = $room_name
                       AND r_item.area_sqft IS NOT NULL
                       AND toFloat(r_item.area_sqft) > toFloat($min_area) |
                 {{ room_name: r_item.name, area_sqft: r_item.area_sqft,
                    length: r_item.length, width: r_item.width }}] }}]
       }} AS answer_data
LIMIT $limit

Key rules:
- ALWAYS include RETURN p, n.name, c.name, dev.name, units, amenities, landmarks PLUS answer_data
- Use > for "greater/larger/bigger", < for "less/smaller/under". Always toFloat() for area comparisons.
- For "at least N sqft" → toFloat(r2.area_sqft) >= toFloat($min_area)
- Use the r.name field for room name matching (canonical names from the ROOM NAMES table above)
- NEVER create a second WITH block that re-aggregates — do it all in one WITH + WHERE EXISTS

4b. ROOM EXISTENCE ("projects with study room"): Same pattern but WHERE only checks r2.name = $room_name (no area filter).
answer_data includes list of matching room details for each unit.

4c. COUNT ("how many projects in Ahmedabad?"):
  WITH count(DISTINCT p) AS total_count, collect(DISTINCT p.project_name) AS project_names
  RETURN {{ count: total_count, projects: project_names }} AS answer_data

4d. COMPARISON ("largest hall?"):
  MATCH (p)-[:HAS_UNIT]->(u)-[:HAS_ROOM]->(r) WHERE r.name = $room_name AND r.area_sqft IS NOT NULL
  WITH p, r, toFloat(r.area_sqft) AS area ORDER BY area DESC LIMIT 10
  Then MATCH location, RETURN project_name, room details, location. LIMIT $limit

=== RULES ===
1. PARAMETERIZE all values with $. Never inline literals.
2. Always "limit": 50 in params. End with LIMIT $limit.
3. answer_columns: ["answer_data"] for LOOKUP/AGGREGATE, [] for GLOBAL/SPECIFIC.
4. vector_query: clean English text for semantic search.
5. Return ONLY raw JSON. No explanation, no markdown, no code fences.
6. LOOKUP: extract pure project name (strip city, "project", "scheme"): "OUM Orbit Ahmedabad" → "OUM ORBIT"
7. CONTAINS (case-insensitive) for project name matching — never exact =.
8. No city filter when project name or neighbourhood is specified.
9. Fence area comparisons: IS NOT NULL AND toFloat() > ...
10. Handle >/</>=/<=/ between. "at least" → >=, "under" → <.
11. Multiple rooms → OR in filter. Prefer MORE data over empty results when ambiguous.
12. ADDRESS + HYPHEN NORM: For ANY location filter, ALWAYS also search p.address AND normalize hyphens:
    (toLower(n.name) CONTAINS toLower($loc) OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower($loc) OR toLower(p.address) CONTAINS toLower($loc))
13. Specific landmark queries: also search p.address.
14. MULTI-LOCATION: When user asks about multiple locations, use SEPARATE params ($nbh_0, $nbh_1 ...) joined with OR. NEVER put them in one comma-joined string.

=== INTENT EXTRACTION ===
- "all projects"/"show all" → query_type="GLOBAL". Filtered → "SPECIFIC".
- "2 bedroom"/"2BHK" → bhk=2. "west Ahmedabad" → zone="West Ahmedabad". "Bopal" → neighbourhood="Bopal".
- "school nearby" → landmark_types=["EDUCATION"]. "near hospital" → ["HEALTHCARE"].
- Fuzzy words → semantic_keywords: "spacious", "luxury", "affordable", "family-friendly".
- If a field is not mentioned, use null or empty list.
- Gujarat localities: Vinzol, Bopal, Nikol, Naroda, Vatva, Gamdi, Gamdi Gaam, Satellite, Chandkheda, Thaltej, Vastrapur, Hanspura, Paldi, Sarkhej, Isanpur, Ghodasar, Kotarpur, Chiloda.
- Sub-locality/road names → neighbourhood. System searches both n.name and p.address.
"""


# ── Fallback Cypher builder ────────────────────────────────────────────────────

def _fallback_cypher(city: Optional[str] = None, bhk: Optional[int] = None, user_query: str = "") -> CypherQuery:
    """
    Minimal fallback Cypher when LLM generation fails.
    Filters by city and/or BHK if available, otherwise returns all.
    Also produces a fallback intent from the raw query.
    """
    match_base = """MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
CALL {
  WITH p
  OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
  WITH u WHERE u IS NOT NULL
  RETURN collect(u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }) AS units
}
CALL {
  WITH p
  OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
  WITH am WHERE am IS NOT NULL
  RETURN collect(am.name) AS amenities
}
CALL {
  WITH p
  OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
  WITH lm WHERE lm IS NOT NULL
  RETURN collect(lm.name) AS landmarks
}
WITH p, n, c, dev, units, amenities, landmarks"""

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
    qt = "SPECIFIC" if wheres else "GLOBAL"
    intent = _extract_fallback_intent(user_query, qt) if user_query else QueryIntent(query_type=qt)

    return CypherQuery(
        cypher=cypher,
        params=params,
        query_type=qt,
        vector_query=f"{bhk} BHK {city or ''}".strip(),
        answer_columns=[],
        intent=intent,
        engine_used="Fallback Rule-based",
    )


# ── Main generation function ───────────────────────────────────────────────────

def generate_cypher(user_query: str, api_keys: dict = None) -> CypherQuery:
    """
    Generate a precise Cypher query AND extract structured intent from the
    user's natural language query in a **single** LLM call.

    Returns a CypherQuery with:
    - cypher: parameterized Cypher ready for Neo4j execution
    - params: parameter dict for the Cypher
    - query_type: "GLOBAL" / "SPECIFIC" / "LOOKUP" / "AGGREGATE"
    - vector_query: text string for ChromaDB semantic search
    - answer_columns: which RETURN columns hold the direct answer
    - intent: QueryIntent parsed from the same LLM response
    - engine_used: String specifying which LLM ran the query

    Falls back to a simple city/BHK filter query if LLM generation fails.
    """
    api_keys = api_keys or {}
    gemini_key = api_keys.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    groq_key = api_keys.get("GROQ_API_KEY") or settings.GROQ_API_KEY

    # ── Try Groq First (fast: ~0.3-0.8s) ──────────────────────────────────────
    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
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
            tokens_used = response.usage.total_tokens if hasattr(response, "usage") and response.usage else 0
            engine_used = f"Groq ({settings.GROQ_MODEL})"

            # Strip accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            cypher = data.get("cypher", "").strip()
            params = data.get("params", {})
            query_type = data.get("query_type", "SPECIFIC")
            vector_query = data.get("vector_query", user_query)
            answer_columns = data.get("answer_columns", [])

            # ── Parse intent from the unified response ──
            intent_data = data.get("intent", {})
            if intent_data and isinstance(intent_data, dict):
                intent_data["query_type"] = query_type
                try:
                    intent = QueryIntent(**intent_data)
                except Exception as ie:
                    logger.warning(f"[Text-to-Cypher] Intent parsing from Groq failed ({ie}). Using minimal intent.")
                    intent = _extract_fallback_intent(user_query, query_type)
            else:
                intent = _extract_fallback_intent(user_query, query_type)

            logger.info(
                f"Intent parsed ({engine_used}): bhk={intent.bhk}, location={intent.neighbourhood or intent.city}, "
                f"amenities={intent.amenities}, type={intent.query_type}"
            )

            if "limit" not in params:
                params["limit"] = 50
            if query_type in ("LOOKUP", "AGGREGATE") and not answer_columns:
                answer_columns = ["answer_data"]
            if not cypher:
                raise ValueError("Groq returned empty cypher")

            cypher = _post_process_cypher(cypher, params)

            logger.info(
                f"[Text-to-Cypher] Engine: {engine_used} | type={query_type} | params={list(params.keys())} | "
                f"answer_cols={answer_columns} | vector_query={vector_query!r}"
            )
            logger.debug(f"[Text-to-Cypher] Cypher:\n{cypher}")

            return CypherQuery(
                cypher=cypher,
                params=params,
                query_type=query_type,
                vector_query=vector_query,
                answer_columns=answer_columns,
                intent=intent,
                tokens_used=tokens_used,
                engine_used=engine_used
            )

        except Exception as e:
            logger.warning(f"[Text-to-Cypher] Groq generation failed ({e}). Falling back to Gemini.")

    # ── Fallback: Gemini (used only when Groq is unavailable/rate-limited) ─────
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=_SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            response = model.generate_content(user_query)
            raw = response.text
            tokens_used = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") and response.usage_metadata else 0
            engine_used = f"Gemini ({settings.GEMINI_MODEL})"

            # Strip accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            cypher = data.get("cypher", "").strip()
            params = data.get("params", {})
            query_type = data.get("query_type", "SPECIFIC")
            vector_query = data.get("vector_query", user_query)
            answer_columns = data.get("answer_columns", [])

            # Parse intent
            intent_data = data.get("intent", {})
            if intent_data and isinstance(intent_data, dict):
                intent_data["query_type"] = query_type
                try:
                    intent = QueryIntent(**intent_data)
                except Exception as ie:
                    logger.warning(f"[Text-to-Cypher] Intent parsing from Gemini failed ({ie}). Using minimal intent.")
                    intent = _extract_fallback_intent(user_query, query_type)
            else:
                intent = _extract_fallback_intent(user_query, query_type)

            logger.info(
                f"Intent parsed ({engine_used}): bhk={intent.bhk}, location={intent.neighbourhood or intent.city}, "
                f"amenities={intent.amenities}, type={intent.query_type}"
            )

            if "limit" not in params:
                params["limit"] = 50
            if query_type in ("LOOKUP", "AGGREGATE") and not answer_columns:
                answer_columns = ["answer_data"]
            if not cypher:
                raise ValueError("Gemini returned empty cypher")

            cypher = _post_process_cypher(cypher, params)

            return CypherQuery(
                cypher=cypher,
                params=params,
                query_type=query_type,
                vector_query=vector_query,
                answer_columns=answer_columns,
                intent=intent,
                tokens_used=tokens_used,
                engine_used=engine_used
            )

        except Exception as e:
            logger.warning(f"[Text-to-Cypher] Gemini generation also failed ({e}). Using minimal fallback.")

    return _fallback_cypher(user_query=user_query)


# ── Known neighbourhood names for fallback extraction ──────────────────────────
_KNOWN_NEIGHBOURHOODS = [
    "Shantigram", "Satellite", "Nikol", "Naroda", "Vinzol", "Bopal", "Vatva",
    "Gamdi", "Gamdi Gaam", "Chandkheda", "Thaltej", "Vastrapur", "Hanspura",
    "Paldi", "Sarkhej", "Isanpur", "Ghodasar", "Kotarpur", "Chiloda",
    "Naranpura", "Kubernagar", "Gokuldham", "Jodhpur",
    "Iscon-Ambli Road", "Science City Road", "New Nikol-Naroda Road",
    "New Nikol - Naroda Road", "Vastral Road", "Naroda, Hanspura Road",
]

# ── Known amenity keywords for fallback extraction ─────────────────────────────
_KNOWN_AMENITY_KEYWORDS = {
    "gym": "gym", "gymnasium": "gym", "fitness": "gym",
    "pool": "swimming pool", "swimming": "swimming pool",
    "clubhouse": "clubhouse", "club house": "clubhouse", "club": "clubhouse",
    "park": "park", "garden": "garden", "lawn": "garden",
    "parking": "parking",
    "jogging": "jogging track", "jogging track": "jogging track",
    "play area": "play area", "playground": "play area",
    "security": "security", "cctv": "cctv", "gated": "gated community",
    "yoga": "yoga", "spa": "spa",
    "library": "library", "theatre": "mini theatre", "theater": "mini theatre",
    "sports": "sports", "cricket": "cricket", "badminton": "badminton",
    "tennis": "tennis", "basketball": "basketball",
    "water supply": "water supply", "power backup": "power backup",
    "lift": "lifts", "elevator": "lifts",
    "piped gas": "piped gas", "smart card": "smart card",
    "commercial": "commercial shops",
}


def _extract_fallback_intent(user_query: str, query_type: str = "SPECIFIC") -> QueryIntent:
    """Extract a robust intent from the raw query when LLM intent parsing fails.

    Scans the query for known neighbourhood names, amenity keywords, BHK numbers,
    and city names so the intent-driven fallback retriever has real filters.
    """
    lower_query = user_query.lower()

    # ── City ──
    fallback_city = None
    for c in ("ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar"):
        if c in lower_query:
            fallback_city = c.title()
            break

    # ── Neighbourhoods — match known names (longest first to avoid partial) ──
    found_nbhs: list[str] = []
    sorted_nbhs = sorted(_KNOWN_NEIGHBOURHOODS, key=len, reverse=True)
    remaining = lower_query
    for nbh in sorted_nbhs:
        if nbh.lower() in remaining:
            found_nbhs.append(nbh)
            # Remove matched text to avoid double-matching substrings
            remaining = remaining.replace(nbh.lower(), " ", 1)
    neighbourhood = found_nbhs if len(found_nbhs) > 1 else (found_nbhs[0] if found_nbhs else None)

    # ── BHK ──
    bhk = None
    bhk_match = re.search(r'(\d)\s*(?:bhk|bed(?:room)?|bedder)', lower_query)
    if bhk_match:
        bhk = int(bhk_match.group(1))

    # ── Property type ──
    property_type = None
    for pt_keyword, pt_value in [("villa", "VILLA"), ("apartment", "APARTMENT"),
                                  ("flat", "APARTMENT"), ("tenement", "TENEMENT"),
                                  ("bungalow", "BUNGALOW"), ("penthouse", "PENTHOUSE")]:
        if pt_keyword in lower_query:
            property_type = pt_value
            break

    # ── Amenities ──
    found_amenities: list[str] = []
    for keyword, canonical in _KNOWN_AMENITY_KEYWORDS.items():
        if keyword in lower_query and canonical not in found_amenities:
            found_amenities.append(canonical)

    # ── Semantic keywords (leftover descriptive words) ──
    semantic = []
    for word in ("spacious", "luxury", "luxurious", "affordable", "big",
                 "lavish", "modern", "premium", "family", "ample"):
        if word in lower_query:
            semantic.append(word)

    return QueryIntent(
        query_type=query_type,
        city=fallback_city,
        neighbourhood=neighbourhood,
        bhk=bhk,
        property_type=property_type,
        amenities=found_amenities if found_amenities else [],
        semantic_keywords=semantic if semantic else [],
    )



def _post_process_cypher(cypher: str, params: dict) -> str:
    """
    Fix common LLM-generated Cypher issues:
    - Ensure toFloat() wraps area_sqft comparisons
    - Fix toLower/toFloat/toInteger applied to list params
    - Expand room name params to synonym groups  (generalized)
    - Expand amenity tag params to synonym groups (generalized)
    """
    # ── 1. Numeric type safety ──────────────────────────────────────────────

    for sqft_field in ('area_sqft', 'carpet_sqft', 'super_builtup_sqft'):
        cypher = re.sub(
            r'(?<!toFloat\()([a-zA-Z]\w*\.' + sqft_field + r')\s*([><!=]+)\s*(?!toFloat)',
            lambda m: f'toFloat({m.group(1)}) {m.group(2)} toFloat',
            cypher,
        )

    # ── 2. Scalar-function-on-list protection ─────────────────────────────────

    # Pattern: IN toLower($param) → IN [x IN $param | toLower(x)]
    cypher = re.sub(
        r'\bIN\s+toLower\(\$(\w+)\)',
        lambda m: f'IN [x IN ${m.group(1)} | toLower(x)]',
        cypher,
    )

    # Generic: for any param that IS a list, fix scalar function calls on it
    for pname, pval in params.items():
        if not isinstance(pval, list):
            continue
        for func in ('toLower', 'toFloat', 'toInteger'):
            pat = re.compile(r'\b' + func + r'\(\$' + re.escape(pname) + r'\)')
            if pat.search(cypher):
                cypher = pat.sub(f'[x IN ${pname} | {func}(x)]', cypher)
                logger.debug(f"[Post-process] Fixed {func} on list param ${pname}")

    # ── 3. Generalized room synonym expansion ─────────────────────────────────
    for pname in list(params.keys()):
        pval = params[pname]
        if 'room' not in pname.lower():
            continue

        if isinstance(pval, str):
            expanded = _expand_room_name(pval)
            if len(expanded) > 1:
                new_pname = pname if pname.endswith('s') else pname + 's'
                params[new_pname] = expanded
                cypher = re.sub(
                    r'(\w+\.name)\s*=\s*\$' + re.escape(pname) + r'\b',
                    r'\1 IN $' + new_pname,
                    cypher,
                )
                cypher = re.sub(
                    r'toLower\((\w+\.name)\)\s*=\s*toLower\(\$' + re.escape(pname) + r'\)',
                    r'\1 IN $' + new_pname,
                    cypher,
                )
                if new_pname != pname and pname in params:
                    del params[pname]
                logger.debug(f"[Post-process] Expanded room '{pval}' -> {expanded}")

        elif isinstance(pval, list):
            all_expanded: list[str] = []
            for name in pval:
                for cn in _expand_room_name(name):
                    if cn not in all_expanded:
                        all_expanded.append(cn)
            if all_expanded != pval:
                params[pname] = all_expanded
                logger.debug(f"[Post-process] Expanded room list -> {all_expanded}")

    # ── 4. Generalized amenity tag synonym expansion ──────────────────────────
    for pname in list(params.keys()):
        pval = params[pname]
        if not ('tag' in pname.lower() or 'amenity' in pname.lower()):
            continue

        if isinstance(pval, str):
            expanded = _expand_amenity_tags(pval)
            if len(expanded) > 1:
                new_pname = 'amenity_tags' if pname == 'tag' else pname
                params[new_pname] = expanded
                cypher = re.sub(
                    r'\$' + re.escape(pname) + r'\s+IN\s+(\w+\.canonical_tags)',
                    r'ANY(t IN $' + new_pname + r' WHERE t IN \1)',
                    cypher,
                )
                if new_pname != pname and new_pname in params:
                    del params[pname]
                logger.debug(f"[Post-process] Expanded amenity '{pval}' -> {expanded}")
            elif len(expanded) == 1 and expanded[0] != pval:
                params[pname] = expanded[0]
                logger.debug(f"[Post-process] Normalized amenity '{pval}' -> '{expanded[0]}'")

        elif isinstance(pval, list):
            all_tags: list[str] = []
            for t in pval:
                for tag in _expand_amenity_tags(t):
                    if tag not in all_tags:
                        all_tags.append(tag)
            if all_tags != pval:
                params[pname] = all_tags
                logger.debug(f"[Post-process] Expanded amenity tags -> {all_tags}")

    # ── 5. Split comma-delimited location/neighbourhood params ────────────────
    # LLM sometimes puts multiple locations into one string: "Shantigram, Satellite, Nikol"
    # Detect this pattern and rewrite into separate OR-based clauses.
    _LOCATION_PARAM_HINTS = ('neighbourhood', 'nbh', 'location', 'loc', 'area', 'road')
    for pname in list(params.keys()):
        pval = params[pname]
        if not isinstance(pval, str):
            continue
        # Only target location-related params
        if not any(hint in pname.lower() for hint in _LOCATION_PARAM_HINTS):
            continue
        # Check if it looks like comma-separated locations (2+ parts)
        parts = [p.strip() for p in pval.split(',') if p.strip()]
        if len(parts) < 2:
            continue

        logger.debug(f"[Post-process] Splitting comma-delimited param ${pname}: {pval!r} -> {parts}")

        # Build OR-based replacement clauses
        new_conds = []
        for i, part in enumerate(parts):
            new_key = f"{pname}_{i}"
            params[new_key] = part
            new_conds.append(
                f"(toLower(n.name) CONTAINS toLower(${new_key}) "
                f"OR toLower(p.address) CONTAINS toLower(${new_key}))"
            )
        or_clause = f"({' OR '.join(new_conds)})"

        # Replace the original CONTAINS clause(s) referencing $pname
        # Pattern 1: (toLower(n.name) CONTAINS toLower($pname) OR toLower(p.address) CONTAINS toLower($pname))
        pattern_full = (
            r'\(\s*toLower\(n\.name\)\s+CONTAINS\s+toLower\(\$' + re.escape(pname) + r'\)'
            r'\s+OR\s+toLower\(p\.address\)\s+CONTAINS\s+toLower\(\$' + re.escape(pname) + r'\)\s*\)'
        )
        if re.search(pattern_full, cypher):
            cypher = re.sub(pattern_full, or_clause, cypher)
        else:
            # Pattern 2: just toLower(n.name) CONTAINS toLower($pname)
            pattern_simple = r'toLower\(n\.name\)\s+CONTAINS\s+toLower\(\$' + re.escape(pname) + r'\)'
            if re.search(pattern_simple, cypher):
                cypher = re.sub(pattern_simple, or_clause, cypher)
            else:
                # Pattern 3: any generic CONTAINS $pname on address or neighbourhood
                pattern_generic = (
                    r'toLower\([a-z]+\.[a-z_]+\)\s+CONTAINS\s+toLower\(\$' + re.escape(pname) + r'\)'
                )
                if re.search(pattern_generic, cypher):
                    cypher = re.sub(pattern_generic, or_clause, cypher)

        # Remove original param
        del params[pname]

    # ── 6. Hyphen-space normalization for n.name CONTAINS ────────────────────
    # Neo4j may store neighbourhoods with spaces around hyphens (e.g. 'New Nikol - Naroda Road')
    # while LLM or user may query without spaces ('New Nikol-Naroda Road').
    # CONTAINS fails in this case. We inject REPLACE(toLower(n.name), ' - ', '-') CONTAINS
    # as an additional OR for every n.name CONTAINS clause that doesn't already have it.
    _nbh_plain = re.compile(
        r'toLower\(n\.name\)\s+CONTAINS\s+toLower\(\$(\w+)\)'
    )
    def _inject_hyphen_norm(m: re.Match) -> str:
        pname = m.group(1)
        original = m.group(0)
        # Only inject if not already wrapped by a REPLACE normalization nearby
        return (
            f"{original} OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower(${pname})"
        )
    # Only apply if REPLACE normalization isn't already present in the cypher
    if "REPLACE(toLower(n.name)" not in cypher:
        cypher = _nbh_plain.sub(_inject_hyphen_norm, cypher)
        if "REPLACE(toLower(n.name)" in cypher:
            logger.debug("[Post-process] Injected hyphen-space normalization for n.name CONTAINS")

    return cypher


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "3 BHK flat in Ahmedabad with pooja room",
        "east-facing villa with swimming pool",
        "ready to move 2 BHK under 1200 sqft carpet area",
        "show me all projects",
        "what is the hall size in OUM Orbit Ahmedabad",
        "kitchen dimensions in Svasaar Pravesh",
        "find me the details of svasaar pravesh",
        "show me all rooms in OUM Orbit",
        "what units are available in OUM Orbit?",
        "what is the address of OUM Orbit?",
        "who is the developer of Svasaar Pravesh?",
        "RERA number of OUM Orbit",
        "does OUM Orbit have swimming pool?",
        "what amenities does Svasaar Pravesh have?",
        "find projects with hall size greater than 100 sqft",
        "find me flats where hall is less than 80 sqft",
        "show 2 BHK apartments with master bedroom bigger than 150 sqft",
        "show me projects that have a study room",
        "find projects with a pooja room",
        "which projects have a servant room",
        "how many projects are in Ahmedabad?",
        "which project has the largest hall?",
        "show all projects by Svasaar",
        "projects near hospital in Ahmedabad",
        "ready to move projects in Ahmedabad",
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
