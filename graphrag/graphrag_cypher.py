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

from graphrag_config import settings, build_gemini_pool, build_groq_pool
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
- Unit {unit_id, project_id, unit_type, property_type (APARTMENT/VILLA/BUNGALOW/ROW_HOUSE/TENEMENT/PENTHOUSE), bhk (integer), entrance_facing (East/West/North/South), carpet_sqft, super_builtup_sqft, balcony_sqft, wash_sqft, num_floors (INTEGER or null — floors this unit spans: Ground-only=1, Ground+First=2, Ground+First+Second=3; null for apartments since floor_level is not set per-room in apartments), applicable_buildings, description}
- Room {room_type, name (canonical string),
           length (RAW STRING — e.g. "10'0\"", "12'-6\"" — NEVER filter on this),
           width  (RAW STRING — NEVER filter on this),
           area_sqft (FLOAT — pre-computed area; use for sqft-based queries: "N sqft", "around N sqft", "> N sqft"),
           length_ft (FLOAT — numeric feet using feet.inches notation: 12'6" → 12.6, 10'0" → 10.0;
                       use ONLY for explicit dimension queries like "10 by 12", "10 x 10"),
           width_ft  (FLOAT — same notation as length_ft; use ONLY for dimension queries),
           floor_level, attached_bathroom (0/1), has_balcony_access (0/1)}
- Amenity {name, category (SPORTS/WELLNESS/SECURITY/NATURE/SOCIAL/INFRASTRUCTURE), canonical_tags (list of strings)}
- Landmark {name, landmark_type (EDUCATION/HEALTHCARE/COMMERCIAL/TRANSPORT/RELIGIOUS/RECREATION)}
- FloorLayout {layout_id, layout_name, total_units_on_floor, has_lifts (0/1/null), has_staircases (0/1/null), corridor_width (string or null), has_refuge_area (0/1/null)}
  NOTE: FloorLayout booleans use 1=confirmed true, null=unknown/not mentioned (treat as not confirmed). Value 0 (explicit false) is rare. For positive checks use = 1. For "does NOT have" checks use (f.field = 0 OR f.field IS NULL).

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
- Room.length, Room.width → RAW STRINGS. NEVER filter on these in Cypher.
- Room.area_sqft → FLOAT. Use for area queries: "N sqft", "around N sqft", "> N sqft". Always toFloat() + IS NOT NULL.
- Room.length_ft, Room.width_ft → FLOAT in feet.inches notation (12'6" → 12.6, 10'0" → 10.0).
  Use ONLY for explicit dimension queries ("10 by 10", "10 x 12", "length 10 width 12").
  Filter: toFloat(r.length_ft) = toFloat($d1) AND toFloat(r.width_ft) = toFloat($d2)
  Always use orientation-agnostic match (10x12 same as 12x10):
    (toFloat(r.length_ft) = toFloat($d1) AND toFloat(r.width_ft) = toFloat($d2))
    OR (toFloat(r.length_ft) = toFloat($d2) AND toFloat(r.width_ft) = toFloat($d1))
  Always fence with IS NOT NULL on both fields.
- carpet_sqft, super_builtup_sqft → FLOAT/NULL. Use toFloat() + IS NOT NULL check.
- bhk → INTEGER. Direct compare: u.bhk = $bhk
- Boolean flags (has_clubhouse etc.) → INTEGER 0/1. For positive check: p.has_pool = 1. For negative check: (p.has_commercial_shops = 0 OR p.has_commercial_shops IS NULL).
- road_widths is a list of mixed strings (e.g., ["30.00 MT.", "12.00M ROAD"]). Extract integer safely: ANY(x IN p.road_widths WHERE toInteger(split(x, '.')[0]) >= toInteger($min_width))
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
    "property_type": <"APARTMENT"|"VILLA"|"TENEMENT"|"BUNGALOW"|"PENTHOUSE"|"ROW_HOUSE"|list|null>,
    "city": <string, list, or null>,
    "neighbourhood": <string, list, or null>,
    "zone": <string, list, or null>,
    "amenities": [<EXTRACT STRICTLY: Do NOT guess amenities from vague words like 'luxurious' or 'peaceful'. Only extract if user explicitly names them (e.g., 'pool', 'gym').>],
    "landmark_types": [<EDUCATION|HEALTHCARE|COMMERCIAL|TRANSPORT|RELIGIOUS|RECREATION>],
    "specific_landmarks": [<place names>],
    "min_sqft": <number or null>, "max_sqft": <number or null>,
    "area_qualifier": <"carpet"|"super_builtup"|null>,
    "around_area": <true|false|null>,
    "min_price_lakhs": <number or null>, "max_price_lakhs": <number or null>,
    "has_balcony": <bool or null>, "has_parking": <bool or null>,
    "entrance_facing": <"East"|"West"|"North"|"South"|null>,
    "developer": <string, list, or null>,
    "project_names": [<project names>],
    "semantic_keywords": [<PUT VAGUE/SUBJECTIVE ADJECTIVES HERE: e.g., 'peaceful', 'luxurious', 'affordable', 'premium'>],
    "min_units_per_floor": <int or null>, "max_units_per_floor": <int or null>,
    "num_floors": <int or null>,
    "room_dimensions": [
      {{"room": "<canonical room name (Bedroom/Kitchen/Hall/etc.)>", "d1": <float feet>, "d2": <float feet>,
        "dim_qualifier": null | "exact" | "around" | "gte" | "lte"}},
      ...
    ]
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
  Pin code (single):   p.pin_code IS NOT NULL AND p.pin_code = $pin_code
  Pin code (multiple): p.pin_code IS NOT NULL AND p.pin_code IN $pin_codes
  (When user gives ONE pin code → use = $pin_code with params: {{ "pin_code": "380007" }})
  (When user gives MULTIPLE pin codes → use IN $pin_codes with params: {{ "pin_codes": ["380007", "380015"] }})
  Room existence (single room): EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r) WHERE r.name = $room_name }}
  Amenity tag:   EXISTS {{ MATCH (p)-[:HAS_AMENITY]->(am2) WHERE $tag IN am2.canonical_tags }}
  Amenity flags: p.has_pool = 1, p.has_clubhouse = 1, (p.has_commercial_shops = 0 OR p.has_commercial_shops IS NULL)

  EMBEDDED ROOMS PATTERN (PREFERRED — use when filtering inside ANY(u IN units WHERE ...)):
  The CALL subquery already embeds rooms into each unit: u.rooms is a list of room property maps.
  Use ANY(r IN u.rooms WHERE ...) to check rooms WITHIN the same unit — simpler and always correct.

  Balcony check (CRITICAL — check BOTH the unit property AND the Balcony room node, since many
  projects store balcony only as a Room called 'Balcony' with u.balcony_sqft = null):
    (u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0)
    OR ANY(r IN u.rooms WHERE r.name IN ['Balcony', 'Terrace'])

  CORRELATED MULTI-ROOM / ROOM+FEATURE filter (ALL conditions on the SAME unit):
  When user wants a unit with MULTIPLE features (e.g. wash area + balcony), use ONE ANY(u IN units WHERE ...) clause:
    Example — "2 BHK with wash area AND balcony":
      ANY(u IN units WHERE
        u.bhk = 2
        AND ANY(r IN u.rooms WHERE r.name IN ['Wash Area'])
        AND (
          (u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0)
          OR ANY(r IN u.rooms WHERE r.name IN ['Balcony', 'Terrace'])
        )
      )
    Example — "wash area and balcony" without BHK:
      ANY(u IN units WHERE
        ANY(r IN u.rooms WHERE r.name IN ['Wash Area'])
        AND (
          (u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0)
          OR ANY(r IN u.rooms WHERE r.name IN ['Balcony', 'Terrace'])
        )
      )
    Example — "2 BHK with pooja room and servant room":
      ANY(u IN units WHERE
        u.bhk = 2
        AND ANY(r IN u.rooms WHERE r.name IN ['Pooja Room'])
        AND ANY(r IN u.rooms WHERE r.name IN ['Servant Room'])
      )
    NEVER split conditions onto separate top-level AND clauses — always keep them inside ONE ANY(u IN units WHERE ...).
    For room checks, ALWAYS use ANY(r IN u.rooms WHERE r.name IN [...]) — do NOT use correlated EXISTS with id().

  Attached bathroom filter (CRITICAL — r.attached_bathroom is stored as 0/1 on each Room node):
  Use ONLY inside the embedded rooms pattern (ANY(r IN u.rooms WHERE ...)), NOT with EXISTS.
  "2 BHK with attached bathroom" — bedroom(s) that have attached_bathroom = 1:
    ANY(u IN units WHERE
      u.bhk = 2
      AND ANY(r IN u.rooms WHERE r.name IN ['Bedroom', 'Master Bedroom'] AND r.attached_bathroom = 1)
    )
  "3 BHK with all bedrooms having attached bathroom" — every bedroom must have one:
    ANY(u IN units WHERE
      u.bhk = 3
      AND ALL(r IN [x IN u.rooms WHERE x.name IN ['Bedroom', 'Master Bedroom']] WHERE r.attached_bathroom = 1)
    )
  "bedroom with balcony access" — similarly uses r.has_balcony_access = 1:
    ANY(u IN units WHERE ANY(r IN u.rooms WHERE r.name IN ['Bedroom', 'Master Bedroom'] AND r.has_balcony_access = 1))
  NOTE: r.attached_bathroom and r.has_balcony_access values: 1 = confirmed yes, 0 or null = no/unknown.
  For "has attached bathroom" always use = 1. For "no attached bathroom" use (r.attached_bathroom = 0 OR r.attached_bathroom IS NULL).

  Floor-level / accessibility filter (for "ground floor bedroom", "senior-friendly", "aging parents"):
    Room.floor_level stores the floor number (0 = ground floor, 1 = first floor, etc.).
    "ground floor bedroom" = a bedroom room at floor_level = 0:
      EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r2) WHERE r2.name IN ['Bedroom','Master Bedroom'] AND toInteger(r2.floor_level) = 0 }}
    Combine with property_type if user specifies (e.g. "ground floor bedroom villa"):
      ANY(u IN units WHERE toLower(u.property_type) = 'villa')
      AND EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r2) WHERE r2.name IN ['Bedroom','Master Bedroom'] AND toInteger(r2.floor_level) = 0 }}
    "aging parents" / "senior-friendly" / "elderly" + villas -> always include ground-floor bedroom check.

  Unit area — ALWAYS search BOTH fields with OR (CRITICAL — many projects store only one field):
  *** MANDATORY RULE: Regardless of whether user says 'carpet' or 'super built-up', ALWAYS use: ***
    ANY(u IN units WHERE
      (u.carpet_sqft IS NOT NULL AND <comparison on toFloat(u.carpet_sqft)>)
      OR (u.super_builtup_sqft IS NOT NULL AND <comparison on toFloat(u.super_builtup_sqft)>))
  This is REQUIRED because many projects only populate one of the two area fields. If you filter
  only on carpet_sqft and the project stores 1372 in super_builtup_sqft (carpet_sqft=null), the
  project will be incorrectly excluded.
  The area_qualifier in intent carries user preference for display only — the Cypher MUST always
  check both fields.
  Area comparisons:
    "around/approximately/roughly/about/~" → >= toFloat($area_min) AND <= toFloat($area_max)  [params: area_min=val*0.85, area_max=val*1.15]
    "exactly N sqft" / plain "N sqft" (no qualifier word) → = toFloat($area_sqft)
    "at least/minimum/>=" → >= toFloat($min_sqft)
    "less than/<" → <= toFloat($max_sqft)
  Facing:        ANY(u IN units WHERE toLower(u.entrance_facing) = toLower($facing))
  Developer:     toLower(dev.name) CONTAINS toLower($developer)
  Property type — SINGLE VALUE (when user names one type explicitly):
    ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))
    Valid values: APARTMENT, VILLA, BUNGALOW, PENTHOUSE, TENEMENT, ROW_HOUSE
    Synonym map (use this to set $property_type or $property_types):
      "flat" / "flats" / "apartment" / "apartments"       → "APARTMENT"
      "villa" / "villas" / "villa type"                   → "VILLA"
      "bungalow" / "bungalows" / "banglow" / "bunglow"    → "BUNGALOW"
      "tenement" / "tenaments" / "tenement type"          → "TENEMENT"
      "pandhout" / "pandhu" / "pandhut"                   → "TENEMENT"  ← Gujarati term
      "row house" / "rowhouse" / "row-house"              → "ROW_HOUSE"
      "penthouse" / "pent house" / "sky villa"            → "PENTHOUSE"
      "house" / "houses" / "ghar" / "makan" / "home" / "homes" /
      "property" / "properties" / "residence" / "unit"   → GENERIC (null — do NOT filter)
  Property type — LIST (when user implies a category of property types):
    ANY(u IN units WHERE toLower(u.property_type) IN [x IN $property_types | toLower(x)])
    Use $property_types (list param) instead of $property_type (string param) in these cases:
      "independent house" / "standalone house" / "independent property"
        → property_types = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"]
      "house with ground floor" / "ground floor house" / "house on ground floor"
        → property_types = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"], no num_floors filter
        (Every independent property has a ground floor; do NOT restrict num_floors)
      "ground floor flat" / "ground floor apartment"
        → property_type = "APARTMENT" (single value; ground floor = building floor)
      "ground floor 2 BHK" / "ground floor unit" (no flat/house keyword)
        → property_types = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"]
      "N floor house" / "N storey house" / "duplex" (duplex = 2 floors)
        → property_types = ["TENEMENT","VILLA","BUNGALOW"], num_floors = N
      "single floor house" / "one floor house"
        → property_types = ["TENEMENT","VILLA","BUNGALOW","ROW_HOUSE"], num_floors = 1
      "house with first floor" → property_types = ["TENEMENT","VILLA","BUNGALOW"], num_floors = 2
      "house with second floor" → property_types = ["TENEMENT","VILLA","BUNGALOW"], num_floors = 3
    NEVER set property_types/property_type for plain "house"/"ghar"/"home" — these are generic.
    NEVER set num_floors for apartments — apartments are single-floor units in a multi-storey building.
  num_floors filter (when user specifies floor count explicitly):
    ANY(u IN units WHERE u.num_floors IS NOT NULL AND toInteger(u.num_floors) = toInteger($num_floors))
    Only use when num_floors is extracted from the query. Do NOT apply for "ground floor" alone.
    NEVER apply to APARTMENT queries.
  Landmark type: EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE lm2.landmark_type = $landmark_type }}
  Specific landmark: (EXISTS {{ MATCH (p)-[:NEAR]->(lm2:Landmark) WHERE toLower(lm2.name) CONTAINS toLower($landmark_name) }} OR toLower(p.address) CONTAINS toLower($landmark_name))
  Total buildings: p.total_buildings IS NOT NULL AND toInteger(p.total_buildings) >= toInteger($min_buildings)

  Floor layout filters (use when user asks about lifts/elevators in building, staircase, refuge area, units per floor):
  IMPORTANT: "lift" queries should PREFER Amenity match (already in amenity tags). Use FloorLayout ONLY when user specifically asks about the building's floor plan having lifts/no-lifts.
  Has lifts in building:     EXISTS {{ MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) WHERE f.has_lifts = 1 }}
  Has staircase in building: EXISTS {{ MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) WHERE f.has_staircases = 1 }}
  Has refuge area:           EXISTS {{ MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) WHERE f.has_refuge_area = 1 }}
  Units per floor (e.g. max 4 units per floor): EXISTS {{ MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) WHERE f.total_units_on_floor IS NOT NULL AND toInteger(f.total_units_on_floor) <= $max_units_per_floor }}
  No lifts (explicit or null treated as no): EXISTS {{ MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) WHERE (f.has_lifts = 0 OR f.has_lifts IS NULL) }}

  Project-level count filters:
  Total apartment blocks (>= N): p.total_buildings IS NOT NULL AND toInteger(p.total_buildings) >= toInteger($min_buildings)
  Total villas (>= N):           p.total_villas IS NOT NULL AND toInteger(p.total_villas) >= toInteger($min_villas)
  (Use these when user asks: "projects with more than 50 villas", "large township with many blocks", etc.)

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
    possession_date: p.project_status, total_buildings: p.total_buildings,
    total_villas: p.total_villas, society_description: p.society_description,
    has_clubhouse: p.has_clubhouse, has_pool: p.has_pool, has_park: p.has_park,
    has_parking: p.has_parking, has_sports_courts: p.has_sports_courts,
    building_names: p.building_names, pin_code: p.pin_code,
    amenities: amenities, landmarks: landmarks
  }}


── TEMPLATE 4: AGGREGATE (numeric comparisons, counts, statistics) ──

4a. ROOM SIZE COMPARISON ("hall > 100 sqft", "bedroom around 150 sqft", "kitchen exactly 80 sqft",
                         "2 BHK with Kitchen around 70 sqft", "master bedroom larger than 120 sqft",
                         "bedroom 10 x 12", "room size 10 by 10", "toilet 4x7", "kitchen length 8 width 9"):

--- ROOM DIMENSION QUERIES ("10 x 12", "10 by 10", "length 10 width 20") ---
When the user gives room DIMENSIONS (not sqft area), use length_ft / width_ft for EXACT matching:

  Step 1 — Identify the room type and the two numbers (in feet).
  Step 2 — DO NOT multiply. DO NOT convert to area. DO NOT use area_sqft.
  Step 3 — Detect qualifier word (if any) in the user query:
    "around/approximately/roughly/about/~/close to/near about" → dim_qualifier="around"
    "at least/minimum/bigger than/larger than/more than/atleast/>=" → dim_qualifier="gte"
    "less than/smaller than/under/<=" → dim_qualifier="lte"
    No qualifier word present → dim_qualifier=null  (exact match, DEFAULT)
  Step 4 — Generate the WHERE clause based on dim_qualifier:

  4a. EXACT (dim_qualifier=null, default — NO qualifier word in query):
    WHERE EXISTS {{
      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
      WHERE r2.name IN $room_names
      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
      AND (
        (toFloat(r2.length_ft) = toFloat($d1) AND toFloat(r2.width_ft) = toFloat($d2))
        OR (toFloat(r2.length_ft) = toFloat($d2) AND toFloat(r2.width_ft) = toFloat($d1))
      )
    }}
    Params: {{d1, d2, room_names}}

  4b. AROUND (dim_qualifier="around" — ±15%% on each dim independently, NOT area):
    Compute: d1_lo=d1*0.85, d1_hi=d1*1.15, d2_lo=d2*0.85, d2_hi=d2*1.15
    WHERE EXISTS {{
      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
      WHERE r2.name IN $room_names
      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
      AND (
        (toFloat(r2.length_ft) >= toFloat($d1_lo) AND toFloat(r2.length_ft) <= toFloat($d1_hi)
         AND toFloat(r2.width_ft) >= toFloat($d2_lo) AND toFloat(r2.width_ft) <= toFloat($d2_hi))
        OR
        (toFloat(r2.length_ft) >= toFloat($d2_lo) AND toFloat(r2.length_ft) <= toFloat($d2_hi)
         AND toFloat(r2.width_ft) >= toFloat($d1_lo) AND toFloat(r2.width_ft) <= toFloat($d1_hi))
      )
    }}
    Params: {{d1_lo, d1_hi, d2_lo, d2_hi, room_names}} (NO d1/d2 raw params for around)
    CRITICAL: ±15%% applies to each dimension INDEPENDENTLY. NEVER multiply d1*d2 to get area.

  4c. GTE (dim_qualifier="gte" — both dims >= given values, orientation-agnostic):
    WHERE EXISTS {{
      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
      WHERE r2.name IN $room_names
      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
      AND (
        (toFloat(r2.length_ft) >= toFloat($d1) AND toFloat(r2.width_ft) >= toFloat($d2))
        OR (toFloat(r2.length_ft) >= toFloat($d2) AND toFloat(r2.width_ft) >= toFloat($d1))
      )
    }}
    Params: {{d1, d2, room_names}}

  4d. LTE (dim_qualifier="lte" — both dims <= given values, orientation-agnostic):
    WHERE EXISTS {{
      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
      WHERE r2.name IN $room_names
      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
      AND (
        (toFloat(r2.length_ft) <= toFloat($d1) AND toFloat(r2.width_ft) <= toFloat($d2))
        OR (toFloat(r2.length_ft) <= toFloat($d2) AND toFloat(r2.width_ft) <= toFloat($d1))
      )
    }}
    Params: {{d1, d2, room_names}}

  Step 5 — query_type=AGGREGATE. Include answer_data with matching_rooms.

  Dimension formats to detect (all handled identically):
    "10x12", "10 x 12", "10 by 12", "10*12", "10\'x12\'", "10 ft by 12 ft"
    "length 10 width 12", "10 length 12 width", "size 10 by 12"
  Assume FEET unless user explicitly says meters/cm/mm.
  Applies to ALL room types: bedroom, master bedroom, kitchen, toilet, bathroom, WC, hall,
    dining, balcony, terrace, wash area, pooja room, store room, study room, servant room,
    dressing room, lobby, passage, courtyard.

  MULTI-ROOM DIMENSION QUERY ("bedroom 10x10, toilet 3x6, balcony 7x4"):
  When user specifies dimensions for MULTIPLE rooms simultaneously, use the EMBEDDED ROOMS
  pattern to ensure ALL rooms are within the SAME unit:
    WHERE ANY(u IN units WHERE
      ANY(r IN u.rooms WHERE r.name IN $room_names_0
        AND r.length_ft IS NOT NULL AND r.width_ft IS NOT NULL
        AND ((toFloat(r.length_ft) = toFloat($d1_0) AND toFloat(r.width_ft) = toFloat($d2_0))
          OR (toFloat(r.length_ft) = toFloat($d2_0) AND toFloat(r.width_ft) = toFloat($d1_0))))
      AND ANY(r IN u.rooms WHERE r.name IN $room_names_1
        AND r.length_ft IS NOT NULL AND r.width_ft IS NOT NULL
        AND ((toFloat(r.length_ft) = toFloat($d1_1) AND toFloat(r.width_ft) = toFloat($d2_1))
          OR (toFloat(r.length_ft) = toFloat($d2_1) AND toFloat(r.width_ft) = toFloat($d1_1))))
    )
  For 3+ rooms, add further AND ANY(...) clauses inside the same ANY(u IN units WHERE ...).
  Params naming: room_names_0, d1_0, d2_0, room_names_1, d1_1, d2_1, room_names_2, d1_2, d2_2 ...
  query_type=SPECIFIC for multi-room dimension queries. answer_columns=[].

CRITICAL DISTINCTION — DO NOT MIX UP:
  "10 x 12 bedroom"  →  DIMENSION query  →  length_ft/width_ft EXACT filter (this section)
                         params: {{d1:10.0, d2:12.0, room_names:[...]}}
                         DO NOT use area_sqft, area_min, area_max
  "bedroom 120 sqft" →  AREA query      →  area_sqft filter (room comparison rules below)
  "bedroom around 120 sqft" → AREA with ±15% → area_min, area_max params
  "2 BHK 900 sqft"   →  UNIT area       →  u.carpet_sqft / u.super_builtup_sqft (Template 2)
                         DO NOT touch r.area_sqft or r.length_ft
  "house 10 x 12"    →  AMBIGUOUS — treat as UNIT area (10*12=120 sqft) → Template 2 SPECIFIC.
                         Only use dimension filter when a specific room type is named.

  The RETURN answer_data matching_rooms comprehension for single-room dimension queries:
    [u_item IN units WHERE u_item IS NOT NULL |
      [r_item IN u_item.rooms WHERE r_item.name IN $room_names
         AND r_item.length_ft IS NOT NULL AND r_item.width_ft IS NOT NULL
         AND ((toFloat(r_item.length_ft) = toFloat($d1) AND toFloat(r_item.width_ft) = toFloat($d2))
           OR (toFloat(r_item.length_ft) = toFloat($d2) AND toFloat(r_item.width_ft) = toFloat($d1)))
       | {{unit_type: u_item.unit_type, bhk: u_item.bhk, room_name: r_item.name,
            area_sqft: r_item.area_sqft, length: r_item.length, width: r_item.width,
            length_ft: r_item.length_ft, width_ft: r_item.width_ft}}]][0]
    }} AS answer_data

Room comparison rules (applied to r2.area_sqft in WHERE EXISTS, and to r_item.area_sqft in RETURN):
  "around/~" / plain N sqft → >= toFloat($area_min) AND toFloat(r2.area_sqft) <= toFloat($area_max)  [area_min=val*0.85, area_max=val*1.15]
  DIMENSION "NxM" / "N by M" → USE length_ft/width_ft exact filter (see DIMENSION section above) — NOT area_sqft
  "exactly N" → = toFloat($area_sqft)
  "at least/bigger/larger/>=" → >= toFloat($min_area)
  "less than/<" → <= toFloat($max_area)
(Room has only ONE area field: area_sqft. No carpet/super distinction for rooms.)
(Room.length and Room.width are STRINGS — NEVER compare them numerically in Cypher.
 Room.length_ft and Room.width_ft are FLOATS — use ONLY for explicit dimension queries, NOT for sqft area filtering.)

CRITICAL: The area filter in the RETURN list comprehension MUST use the SAME
parameters and comparison operator as in the WHERE EXISTS clause.
  "around" → BOTH WHERE EXISTS and RETURN use: >= toFloat($area_min) AND toFloat(r_item.area_sqft) <= toFloat($area_max)
  "larger than" → BOTH use: > toFloat($min_area)

CRITICAL: matching_rooms must be a FLAT list of room objects (not unit-wrapped).
Each item must contain: unit_type, bhk, room_name, area_sqft, length, width.

Use this EXACT structure for "around" (+/- 15% range) queries:

MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
WHERE EXISTS {{
  MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
  WHERE r2.name = $room_name AND r2.area_sqft IS NOT NULL
  AND toFloat(r2.area_sqft) >= toFloat($area_min) AND toFloat(r2.area_sqft) <= toFloat($area_max)
}}
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
RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
       units, amenities, landmarks,
       {{ project_name: p.project_name, city: c.name, neighbourhood: n.name,
          matching_rooms: [u_item IN units WHERE u_item IS NOT NULL |
            [r_item IN u_item.rooms WHERE r_item.name = $room_name
             AND r_item.area_sqft IS NOT NULL
             AND toFloat(r_item.area_sqft) >= toFloat($area_min) AND toFloat(r_item.area_sqft) <= toFloat($area_max) |
               {{ unit_type: u_item.unit_type, bhk: u_item.bhk,
                  room_name: r_item.name, area_sqft: r_item.area_sqft,
                  length: r_item.length, width: r_item.width }}]][0]
       }} AS answer_data
LIMIT $limit

For "larger than / greater than / >=" (non-around) queries, use this RETURN comprehension:
       {{ project_name: p.project_name, city: c.name, neighbourhood: n.name,
          matching_rooms: [u_item IN units WHERE u_item IS NOT NULL |
            [r_item IN u_item.rooms WHERE r_item.name = $room_name
             AND r_item.area_sqft IS NOT NULL
             AND toFloat(r_item.area_sqft) > toFloat($min_area) |
               {{ unit_type: u_item.unit_type, bhk: u_item.bhk,
                  room_name: r_item.name, area_sqft: r_item.area_sqft,
                  length: r_item.length, width: r_item.width }}]][0]
       }} AS answer_data

If the query also specifies BHK (e.g. "2 BHK with Kitchen around 70 sqft"), add BHK filter:
  WHERE r2.name = $room_name AND u2.bhk = $bhk AND r2.area_sqft IS NOT NULL AND ...
  AND in the RETURN comprehension: WHERE u_item IS NOT NULL AND u_item.bhk = $bhk
Note: [list][0] flattens nested list — if no match it returns null (safely skipped by the reader).

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
10. Handle >/</>=/<= between. "at least" → >=, "under" → <.
11. Multiple rooms → OR in filter. For room existence queries, do NOT drop property_type filter to get more results — the user's type is always a hard constraint.
12. ADDRESS + HYPHEN NORM: For ANY location filter, ALWAYS also search p.address AND normalize hyphens:
    (toLower(n.name) CONTAINS toLower($loc) OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower($loc) OR toLower(p.address) CONTAINS toLower($loc))
13. Specific landmark queries: also search p.address.
14. MULTI-LOCATION: When user asks about multiple locations, use SEPARATE params ($nbh_0, $nbh_1 ...) joined with OR. NEVER put them in one comma-joined string.
15. PROPERTY TYPE IS A HARD FILTER: If the user says "apartment", "villa", "penthouse", "tenement", "row house", or "bungalow", that word is NEVER a semantic_keyword — it is ALWAYS a property_type filter in the Cypher WHERE clause AND in intent.property_type.
16. "AROUND" QUERIES: If the user provides a bare area number (e.g. "900 sqft") OR uses "around"/"approximately"/"roughly"/"about"/"~"/"close to" → use >= AND <= window.
    area_min = val * 0.85, area_max = val * 1.15 — include BOTH in params.
    Example: "900 sqft" or "around 900 sqft" → params: area_min=765.0, area_max=1035.0 → >= toFloat($area_min) AND toFloat(field) <= toFloat($area_max).
    NEVER use one-sided >= for "around" or plain numbers. Same rule for room area_sqft queries.
17. AREA FIELD SELECTION (unit level):
    - No qualifier (user says just "area" or "sqft") → match EITHER field with OR:
      (u.carpet_sqft IS NOT NULL AND ...) OR (u.super_builtup_sqft IS NOT NULL AND ...)
    - "carpet area" / "carpet sqft" / "by carpet" → STILL use both fields with OR (carpet_sqft OR super_builtup_sqft). Record area_qualifier=carpet in intent for display.
    - "super built-up" / "super builtup" / "built-up area" / "SBA" → STILL use both fields with OR. Record area_qualifier=super_builtup in intent for display.
    - "exactly N sqft" → = toFloat($area_sqft), still apply both fields with OR.
    - Rooms only have area_sqft — field selection does not apply to rooms, only comparison logic.
18. SAME-UNIT CORRELATION (CRITICAL): When user wants a unit that has MULTIPLE features simultaneously
    (e.g. BHK + room type + balcony), ALL conditions MUST be in ONE ANY(u IN units WHERE ...) clause.
    NEVER split into separate top-level AND clauses.
    For room checks within the same unit, use ANY(r IN u.rooms WHERE r.name IN [...]) — NOT correlated EXISTS.
    Balcony check must be: (u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0) OR ANY(r IN u.rooms WHERE r.name IN ['Balcony','Terrace'])
    Wrong:  ANY(u IN units WHERE u.bhk=2) AND EXISTS{{...wash area...}} AND ANY(u IN units WHERE u.balcony_sqft>0)
    Correct: ANY(u IN units WHERE u.bhk=2 AND ANY(r IN u.rooms WHERE r.name IN ['Wash Area']) AND ((u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft)>0) OR ANY(r IN u.rooms WHERE r.name IN ['Balcony','Terrace'])))
19. GROUND FLOOR / ACCESSIBILITY QUERIES: Phrases like "ground floor bedroom", "bedroom on ground floor",
    "suitable for aging parents", "elderly", "senior-friendly villa" → the key structural signal is a bedroom
    at floor_level=0. Always generate:
      EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2)-[:HAS_ROOM]->(r2) WHERE r2.name IN ['Bedroom','Master Bedroom'] AND toInteger(r2.floor_level) = 0 }}
    Combined with property_type if user specified one (villa, tenement, etc.).
    Also set intent.semantic_keywords=["ground floor bedroom", "senior-friendly"] for vector fallback.
20. ROOM DIMENSION QUERIES ("bedroom 10 x 12", "toilet 4 by 7", "kitchen 8x10", "room size 10 by 10",
    "length 10 width 20", "10'x12' bedroom", "washroom 4 ft by 6 ft",
    "bedroom 10x10, toilet 3x6, balcony 7x4"):
    a) Identify the ROOM TYPE(S) from room names (use ROOM NAMES table for canonical names).
    b) Extract the dimension numbers as d1 and d2 (in feet). DO NOT multiply. DO NOT compute area.
    c) Use Room.length_ft and Room.width_ft with EXACT orientation-agnostic match (see DIMENSION section).
       Single room:  params {{d1, d2, room_names}}  with EXISTS subquery on length_ft/width_ft.
       Multi-room:   params {{d1_0, d2_0, room_names_0, d1_1, d2_1, room_names_1, ...}}
                     with ANY(u IN units WHERE ...) embedded rooms pattern (same-unit guarantee).
    d) NEVER use area_sqft, area_min, area_max for dimension queries.
    e) NEVER filter r.length or r.width (raw strings, not numbers).
    f) NEVER put dimension values into u.carpet_sqft / u.super_builtup_sqft.
    g) Single room dimension  → query_type=AGGREGATE, answer_columns=["answer_data"].
       Multi-room dimensions  → query_type=SPECIFIC, answer_columns=[].
    h) *** COMBINED: unit area + room dimension (MOST COMMON REAL-WORLD CASE) ***
       If user mentions BOTH a unit area (carpet sqft, super built-up) AND a room dimension:
       Examples: "1372 carpet sqft with bedroom 10.6 by 12.4"
                 "2 BHK around 900 sqft, bedroom 10x12"
                 "flat 1200 sqft with kitchen 8 by 10 and gym"
       MANDATORY rules:
         i)  query_type = "SPECIFIC"  (NEVER AGGREGATE for these)
         ii) answer_columns = []
         iii) Use Template 2 (BASE MATCH + WHERE after WITH) with:
              - Unit area filter using BOTH fields with OR (see rule 17 above).
              - PLUS room dimension: AND EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
                                       WHERE r2.name IN $room_names
                                       AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
                                       AND ((toFloat(r2.length_ft) = toFloat($d1) AND toFloat(r2.width_ft) = toFloat($d2))
                                         OR (toFloat(r2.length_ft) = toFloat($d2) AND toFloat(r2.width_ft) = toFloat($d1))) }}
              - Any other filters (BHK, amenity, location, landmark) are added as AND clauses too.
         iv) DO NOT use answer_data, matching_rooms, or any AGGREGATE RETURN structure.
         v)  The full WHERE block example:
             WHERE ANY(u IN units WHERE
               (u.carpet_sqft IS NOT NULL AND toFloat(u.carpet_sqft) >= toFloat($area_min) AND toFloat(u.carpet_sqft) <= toFloat($area_max))
               OR (u.super_builtup_sqft IS NOT NULL AND toFloat(u.super_builtup_sqft) >= toFloat($area_min) AND toFloat(u.super_builtup_sqft) <= toFloat($area_max)))
             AND EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)
               WHERE r2.name IN $room_names
               AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL
               AND ((toFloat(r2.length_ft) = toFloat($d1) AND toFloat(r2.width_ft) = toFloat($d2))
                 OR (toFloat(r2.length_ft) = toFloat($d2) AND toFloat(r2.width_ft) = toFloat($d1))) }}
         INTENT fields for this case:
           intent.min_sqft = <target value>  (if "around" → also set intent.around_area=true)
           intent.area_qualifier = "carpet" | "super_builtup" | null  (display only)
           intent.room_dimensions = [{{"room": "Bedroom", "d1": 10.6, "d2": 12.4, "dim_qualifier": null}}]
           # "around bedroom 10 by 12": dim_qualifier="around" → ±15%% on each dim
           # "bedroom at least 10 by 12": dim_qualifier="gte"
           # "bedroom less than 10 by 12": dim_qualifier="lte"

=== INTENT EXTRACTION ===
- "all projects"/"show all" → query_type="GLOBAL". Filtered → "SPECIFIC".
- "2 bedroom"/"2BHK" → bhk=2. "west Ahmedabad" → zone="West Ahmedabad". "Bopal" → neighbourhood="Bopal".
- "school nearby" → landmark_types=["EDUCATION"]. "near hospital" → ["HEALTHCARE"].
- Fuzzy words → semantic_keywords: "spacious", "luxury", "affordable", "family-friendly".
- PROPERTY TYPE (hard structural filter, NEVER put in semantic_keywords):
    "apartment" / "flat" / "flats" → property_type="APARTMENT"
    "villa" / "villas" → property_type="VILLA"
    "penthouse" / "pent house" / "sky villa" → property_type="PENTHOUSE"
    "row house" / "rowhouse" → property_type="ROW_HOUSE"
    "tenement" → property_type="TENEMENT"
    "bungalow" → property_type="BUNGALOW"
    "house" / "houses" / "homes" / "home" / "property" / "properties" / "residence" / "unit" → GENERIC — set property_type=null. These are informal synonyms for ANY residential property. Do NOT map to VILLA or any other type.
  Whenever property_type is set in intent, the Cypher WHERE clause MUST include:
    ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))
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
CALL (p) {
  OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
  WITH u WHERE u IS NOT NULL
  RETURN collect(u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }) AS units
}
CALL (p) {
  OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
  WITH am WHERE am IS NOT NULL
  RETURN collect(am.name) AS amenities
}
CALL (p) {
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

    Primary:  Gemini (pool of up to 4 keys, round-robin with auto-fallback)
    Fallback: Groq   (pool of up to 2 keys, round-robin with auto-fallback)

    If a key hits a rate-limit or fails, the next key in the pool is tried
    automatically. If all keys in both pools fail, returns a rule-based fallback.
    """
    api_keys = api_keys or {}
    gemini_pool = build_gemini_pool(api_keys)
    groq_pool   = build_groq_pool(api_keys)

    def _parse_llm_response(raw: str, engine_used: str, tokens_used: int) -> CypherQuery:
        """Parse the raw JSON string from any LLM into a CypherQuery."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        cypher       = data.get("cypher", "").strip()
        params       = data.get("params", {})
        query_type   = data.get("query_type", "SPECIFIC")
        vector_query = data.get("vector_query", user_query)
        answer_columns = data.get("answer_columns", [])

        intent_data = data.get("intent", {})
        if intent_data and isinstance(intent_data, dict):
            intent_data["query_type"] = query_type
            try:
                intent = QueryIntent(**intent_data)
            except Exception as ie:
                logger.warning(f"[Text-to-Cypher] Intent parsing failed ({ie}). Using fallback intent.")
                intent = _extract_fallback_intent(user_query, query_type)
        else:
            intent = _extract_fallback_intent(user_query, query_type)

        logger.info(
            f"Intent parsed ({engine_used}): bhk={intent.bhk}, "
            f"location={intent.neighbourhood or intent.city}, "
            f"amenities={intent.amenities}, type={intent.query_type}"
        )

        if "limit" not in params:
            params["limit"] = 50
        if query_type in ("LOOKUP", "AGGREGATE") and not answer_columns:
            answer_columns = ["answer_data"]
        if not cypher:
            raise ValueError(f"{engine_used} returned empty cypher")

        cypher = _post_process_cypher(cypher, params, user_query)
        logger.info(
            f"[Text-to-Cypher] Engine: {engine_used} | type={query_type} | "
            f"params={list(params.keys())} | answer_cols={answer_columns} | "
            f"vector_query={vector_query!r}"
        )
        logger.debug(f"[Text-to-Cypher] Cypher:\n{cypher}")

        return CypherQuery(
            cypher=cypher, params=params, query_type=query_type,
            vector_query=vector_query, answer_columns=answer_columns,
            intent=intent, tokens_used=tokens_used, engine_used=engine_used,
        )

    # ── Primary: Gemini pool ───────────────────────────────────────────────────
    n_gemini = len(gemini_pool)
    for attempt in range(n_gemini):
        key = gemini_pool.next()
        if not key:
            break
        key_slot = gemini_pool.last_slot
        logger.info(f"[Text-to-Cypher] Gemini key slot {key_slot}/{n_gemini} (retry attempt {attempt + 1})")
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction=_SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            response = model.generate_content(user_query)
            raw = response.text
            tokens_used = (
                response.usage_metadata.total_token_count
                if hasattr(response, "usage_metadata") and response.usage_metadata else 0
            )
            return _parse_llm_response(raw, f"Gemini ({settings.GEMINI_MODEL})", tokens_used)
        except Exception as e:
            logger.warning(
                f"[Text-to-Cypher] Gemini slot {key_slot} failed ({e}). "
                f"{'Trying next Gemini key.' if attempt + 1 < n_gemini else 'All Gemini keys exhausted.'}"
            )

    # ── Fallback: Groq pool ────────────────────────────────────────────────────
    n_groq = len(groq_pool)
    if n_groq:
        logger.warning("[Text-to-Cypher] Falling back to Groq pool.")
    for attempt in range(n_groq):
        key = groq_pool.next()
        if not key:
            break
        key_slot = groq_pool.last_slot
        logger.info(f"[Text-to-Cypher] Groq key slot {key_slot}/{n_groq} (retry attempt {attempt + 1})")
        try:
            groq_client = Groq(api_key=key)
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_query},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            tokens_used = (
                response.usage.total_tokens
                if hasattr(response, "usage") and response.usage else 0
            )
            return _parse_llm_response(raw, f"Groq ({settings.GROQ_MODEL})", tokens_used)
        except Exception as e:
            logger.warning(
                f"[Text-to-Cypher] Groq slot {key_slot} failed ({e}). "
                f"{'Trying next Groq key.' if attempt + 1 < n_groq else 'All Groq keys exhausted.'}"
            )

    logger.warning("[Text-to-Cypher] All LLM keys exhausted. Using rule-based fallback.")
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
    city names, sqft area, and room dimensions so the intent-driven fallback retriever
    has real filters even when the LLM is unavailable.
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
                                  ("bungalow", "BUNGALOW"), ("penthouse", "PENTHOUSE"),
                                  ("row house", "ROW_HOUSE"), ("rowhouse", "ROW_HOUSE")]:
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

    # ── Sqft area ──
    min_sqft = None
    max_sqft = None
    area_qualifier = None
    around_area = None

    # Detect area qualifier
    if any(w in lower_query for w in ("carpet area", "carpet sqft", "carpet")):
        area_qualifier = "carpet"
    elif any(w in lower_query for w in ("super built-up", "super builtup", "built-up area", "sba")):
        area_qualifier = "super_builtup"

    # Detect "around" / approximate marker
    if any(w in lower_query for w in ("around", "approximately", "roughly", "about", "approx", "close to", "near about")):
        around_area = True

    # Extract sqft value  e.g. "1372 sqft", "1372 carpet sqft", "1372 super builtup sqft"
    # Allow 0-3 words between number and sqft keyword
    sqft_match = re.search(
        r'(\d[\d,\.]*)(?:\s+\w+){0,3}?\s*(?:sq\.?\s*ft|sqft|sq\.ft)',
        lower_query
    )

    if sqft_match:
        try:
            val = float(sqft_match.group(1).replace(",", ""))
            if around_area:
                min_sqft = round(val * 0.85, 2)
                max_sqft = round(val * 1.15, 2)
            else:
                min_sqft = val
        except ValueError:
            pass

    # ── Room dimensions ──
    # Regex patterns: "bedroom 10.6 by 12.4", "kitchen 8x10", "hall 10 x 10", "toilet 4*7"
    # Map common room keywords to canonical names
    _ROOM_KEYWORD_MAP = [
        (r"master\s+bed(?:room)?", "Master Bedroom"),
        (r"bed\s*room|bedroom",    "Bedroom"),
        (r"kitchen|kitchenette",   "Kitchen"),
        (r"hall|living|drawing\s*room|lounge", "Hall"),
        (r"dining(?:\s*room)?",    "Dining"),
        (r"toilet|wash\s*room|bathroom|wc|restroom", "Toilet"),
        (r"balcony",               "Balcony"),
        (r"terrace|rooftop",       "Terrace"),
        (r"wash\s*area|utility|laundry", "Wash Area"),
        (r"pooja\s*room|puja(?:\s*room)?|mandir|prayer\s*room", "Pooja Room"),
        (r"store\s*room|storage(?:\s*room)?", "Store Room"),
        (r"study\s*room|study",    "Study Room"),
        (r"servant\s*room|maid\s*room", "Servant Room"),
        (r"dressing\s*room|walk-in\s*closet", "Dressing Room"),
        (r"lobby|foyer",           "Lobby"),
        (r"passage|corridor|hallway", "Passage"),
    ]
    _DIM_PATTERN = r'(\d+\.?\d*)\s*(?:by|x|\*|×|ft\s*x|ft\s*by)\s*(\d+\.?\d*)'

    room_dimensions: list[dict] = []
    seen_canonicals: set[str] = set()
    for room_pat, canonical in _ROOM_KEYWORD_MAP:
        if canonical in seen_canonicals:
            continue
        # Forward: "<room keyword> ... <d1> by/x <d2>"
        fwd = re.search(
            r'(?:' + room_pat + r')(?:\s+\w+){0,4}?\s+' + _DIM_PATTERN,
            lower_query
        )
        # Backward: "<d1> by/x <d2> ... <room keyword>"
        bwd = re.search(
            _DIM_PATTERN + r'\s+(?:\w+\s+){0,3}?' + r'(?:' + room_pat + r')',
            lower_query
        )
        match = fwd or bwd
        if match:
            try:
                if fwd:
                    d1_str, d2_str = fwd.group(1), fwd.group(2)
                else:
                    d1_str, d2_str = bwd.group(1), bwd.group(2)
                if d1_str is None or d2_str is None:
                    continue
                d1, d2 = float(d1_str), float(d2_str)
                # Sanity check: room dimensions should be reasonable (1–100 ft)
                if 1.0 <= d1 <= 100.0 and 1.0 <= d2 <= 100.0:
                    # Detect qualifier word near this dimension match
                    _prefix = lower_query[:match.end()]
                    _dim_qualifier = None
                    _AROUND = ('around', 'approximately', 'roughly', 'about', 'approx', 'close to', 'near about', '~')
                    _GTE    = ('at least', 'atleast', 'minimum', 'bigger than', 'larger than', 'more than', 'at-least')
                    _LTE    = ('less than', 'smaller than', 'under', 'no more than')
                    if any(w in _prefix for w in _AROUND):
                        _dim_qualifier = 'around'
                    elif any(w in _prefix for w in _GTE):
                        _dim_qualifier = 'gte'
                    elif any(w in _prefix for w in _LTE):
                        _dim_qualifier = 'lte'
                    room_dimensions.append({"room": canonical, "d1": d1, "d2": d2, "dim_qualifier": _dim_qualifier})
                    seen_canonicals.add(canonical)
            except (ValueError, IndexError, TypeError):
                pass

    return QueryIntent(
        query_type=query_type,
        city=fallback_city,
        neighbourhood=neighbourhood,
        bhk=bhk,
        property_type=property_type,
        amenities=found_amenities if found_amenities else [],
        semantic_keywords=semantic if semantic else [],
        min_sqft=min_sqft,
        max_sqft=max_sqft,
        area_qualifier=area_qualifier,
        around_area=around_area,
        room_dimensions=room_dimensions if room_dimensions else [],
    )




def _post_process_cypher(cypher: str, params: dict, user_query: str = "") -> str:
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
                # Rewrite ALL occurrences of .name = $pname in the entire Cypher
                # (covers both WHERE EXISTS subquery AND RETURN list comprehensions)
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
                # Also fix any remaining bare $pname references (e.g. $room_name used
                # as a plain string value inside comprehensions after = was rewritten)
                # These are safe to leave as-is since we keep params[new_pname]=list.
                # However, if any reference to $pname (the deleted key) still exists
                # as a non-.name context (e.g. room_name: $room_name in RETURN map),
                # rewrite those to $new_pname so Neo4j receives a valid param.
                if new_pname != pname:
                    cypher = re.sub(
                        r'\$' + re.escape(pname) + r'\b',
                        '$' + new_pname,
                        cypher,
                    )
                    if pname in params:
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

    # ── 7. "Around" safety net — patch one-sided >= / = into >= X AND <= Y ─────
    _AROUND_WORDS = {"around", "approximately", "roughly", "about", "approx",
                     "close to", "near about"}
    _is_around = user_query and any(w in user_query.lower() for w in _AROUND_WORDS)
    _is_exact = user_query and any(w in user_query.lower() for w in {"exactly", "exact", "precisely"})

    _SOLO_CHECKS = []
    if not _is_exact:
        # If user didn't say "exactly", we expand any exact matches (= area_sqft) into bands
        _SOLO_CHECKS.append(("area_sqft", None, "area_min", "area_max"))
        
    if _is_around:
        # If user explicitly said "around" but LLM still generated a one-sided bound, patch it
        _SOLO_CHECKS.extend([
            ("min_sqft",  "max_sqft",  "area_min", "area_max"),
            ("min_area",  "max_area",  "area_min", "area_max"),
        ])

    if _SOLO_CHECKS:
        _AREA_TOLERANCE = 0.15
        for solo_key, max_key, lo_key, hi_key in _SOLO_CHECKS:
            if solo_key not in params:
                continue
            if max_key and max_key in params:
                continue  # LLM already gave both bounds — respect it
            if lo_key in params and hi_key in params:
                continue  # Already patched
            val = params[solo_key]
            if not isinstance(val, (int, float)):
                continue
            lo = round(val * (1 - _AREA_TOLERANCE), 2)
            hi = round(val * (1 + _AREA_TOLERANCE), 2)
            params[lo_key] = lo
            params[hi_key] = hi
            # Rewrite '>= toFloat($solo_key)' → '>= toFloat($lo_key) AND ... <= toFloat($hi_key)'
            cypher = re.sub(
                r'(toFloat\([^)]+\))\s*>=\s*toFloat\(\$' + re.escape(solo_key) + r'\)',
                rf'\1 >= toFloat(${lo_key}) AND \1 <= toFloat(${hi_key})',
                cypher,
            )
            # Rewrite '= toFloat($solo_key)' (exact match) → >= AND <=
            cypher = re.sub(
                r'(toFloat\([^)]+\))\s*=\s*toFloat\(\$' + re.escape(solo_key) + r'\)',
                rf'\1 >= toFloat(${lo_key}) AND \1 <= toFloat(${hi_key})',
                cypher,
            )
            if solo_key in params:
                del params[solo_key]
            logger.debug(
                f"[Post-process] 'Around' safety net: ${solo_key}={val} "
                f"→ >= {lo} AND <= {hi}"
            )
            break  # Only patch the first matching solo param

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
