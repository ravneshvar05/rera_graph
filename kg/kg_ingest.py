"""
kg_ingest.py — Production batch ingestion pipeline.

Reads every brochure JSON from the output/ folder and writes:
  1. Neo4j  → full knowledge graph (Project, Unit, Room, Landmark, Amenity,
              City, Neighbourhood, Zone, Developer nodes + all relationships)
  2. ChromaDB → one embedding document per unit (rich text, ALL fields combined)

Design principles:
  - Zero data loss: every JSON field is stored somewhere in the graph
  - Idempotent: MERGE everywhere except Room (rooms belong to one unit)
  - Null-safe: literal "null"/"none" strings treated as None
  - Parallel: ThreadPoolExecutor (configurable workers), Neo4j driver is thread-safe
  - Progress: logs every 100 files with rate and ETA

Usage:
    python kg_ingest.py                    # ingest new files only
    python kg_ingest.py --force            # wipe graph and ChromaDB, re-ingest all
    python kg_ingest.py --dry-run          # validate JSONs, no writes
    python kg_ingest.py --workers 16       # override worker count
    python kg_ingest.py --dir ./output     # override JSON directory
"""

import argparse
import json
import re as _re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from kg_config import settings

# ══════════════════════════════════════════════════════════════════════════════
#  NULL / SAFE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_NULL_STRINGS: frozenset[str] = frozenset(
    {"null", "none", "na", "n/a", "", "undefined", "unknown", "nil"}
)


def _clean(val: Any) -> Any:
    """Return None for null-like values; otherwise return the value as-is."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in _NULL_STRINGS:
        return None
    return val


def _str(val: Any) -> Optional[str]:
    """Clean and return as stripped string, or None."""
    v = _clean(val)
    return str(v).strip() if v is not None else None


def _float(val: Any) -> Optional[float]:
    """Clean and return as float, or None."""
    v = _clean(val)
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int(val: Any) -> Optional[int]:
    """Clean and return as int, or None."""
    v = _clean(val)
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _bool_int(val: Any) -> Optional[int]:
    """Return 1/0/None for boolean fields."""
    v = _clean(val)
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return 1 if v else 0
    s = str(v).lower()
    if s in ("true", "yes", "1"):
        return 1
    if s in ("false", "no", "0"):
        return 0
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT ID NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_project_id(raw_id: Any, project_name: str, source_file: str) -> str:
    """
    Return a stable, unique project_id.
    - If raw_id is a real value (not null/empty/literal-"null"), use it.
    - Otherwise build a deterministic slug from project_name + source filename.
    """
    cleaned = _str(raw_id)
    if cleaned:
        return cleaned
    name_part = _re.sub(r"[^\w]+", "_", (project_name or "unknown").strip())[:40]
    file_part = _re.sub(r"[^\w]+", "_", Path(source_file).stem)[:20]
    return f"{name_part}__{file_part}"


# ══════════════════════════════════════════════════════════════════════════════
#  CITY NORMALISATION  (directly ported from root ingest.py, extended)
# ══════════════════════════════════════════════════════════════════════════════

_GUJARATI_CITY_MAP: dict[str, str] = {
    "અમદાવાદ": "Ahmedabad", "સુરત": "Surat",    "વડોદરા": "Vadodara",
    "રાજકોટ":  "Rajkot",    "ભાવનગર": "Bhavnagar","ગાંધીનગર": "Gandhinagar",
    "જામનગર": "Jamnagar",   "જૂનાગઢ": "Junagadh", "આણંદ": "Anand",
    "નડિયાદ":  "Nadiad",    "મહેસાણા": "Mehsana", "પોરબંદર": "Porbandar",
    "મોરબી":  "Morbi",      "ભૂજ": "Bhuj",        "વલસાડ": "Valsad",
    "ગાંઘ":   "Gandhinagar",
}

_AHMEDABAD_LOCALITIES: frozenset[str] = frozenset({
    "isanpur", "nikol", "new nikol", "naroda", "new naroda",
    "bopal", "new bopal", "sarkhej", "thaltej", "prahlad nagar",
    "satellite", "bodakdev", "vastrapur", "maninagar", "ghatlodia",
    "chandkheda", "ranip", "vastral road", "vastrapur road",
    "science city road", "sg highway", "sg road", "shela",
    "shantigram", "south bopal", "ambli", "shilaj", "vejalpur",
    "vinzol", "vizol", "hanspur", "hanspura", "gamdi", "gamdi gaam",
    "chiloda", "kotarpur", "chiloda-kotarpur", "gokuldham", "vatva",
    "narol", "odhav", "hathijan", "vastral", "kathwada",
})

_SURAT_LOCALITIES: frozenset[str] = frozenset({
    "adajan", "pal", "sarthana", "katargam", "varachha",
    "vesu", "piplod", "jahangirpura",
})

_LOCALITY_TO_CITY: dict[str, str] = {
    **{loc: "Ahmedabad" for loc in _AHMEDABAD_LOCALITIES},
    **{loc: "Surat"     for loc in _SURAT_LOCALITIES},
}


def normalise_city(raw_city: Any, neighbourhood: Any = None) -> Optional[str]:
    """Normalise city name: Gujarati script, ALL-CAPS, locality-as-city."""
    raw = _str(raw_city)
    if raw:
        if raw in _GUJARATI_CITY_MAP:
            return _GUJARATI_CITY_MAP[raw]
        city_clean = raw.strip()
        if city_clean.isupper():
            city_clean = city_clean.title()
        if city_clean.lower() in _LOCALITY_TO_CITY:
            return _LOCALITY_TO_CITY[city_clean.lower()]
        return city_clean
    nbhd = _str(neighbourhood)
    if nbhd and nbhd.lower() in _LOCALITY_TO_CITY:
        return _LOCALITY_TO_CITY[nbhd.lower()]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION HELPERS  (written at ingest time — never at query time)
# ══════════════════════════════════════════════════════════════════════════════

_AMENITY_KEYWORDS: dict[str, set[str]] = {
    "SPORTS":    {"gym", "gymnasium", "swimming", "pool", "jogging", "running",
                  "sports", "cricket", "volleyball", "basketball", "badminton",
                  "tennis", "indoor sports", "outdoor sports"},
    "WELLNESS":  {"health club", "health", "spa", "yoga", "meditation",
                  "mini theatre", "theatre", "theater"},
    "SECURITY":  {"security", "cctv", "gated", "guard", "surveillance",
                  "security cabin", "security system"},
    "NATURE":    {"garden", "park", "landscap", "green", "tree", "lawn",
                  "courtyard", "fountain", "common plot"},
    "SOCIAL":    {"clubhouse", "club house", "lounge", "party", "celebration",
                  "community", "amphitheatre", "children play", "kids play",
                  "senior citizen", "chess", "carom", "card", "library",
                  "indoor games"},
    "INFRASTRUCTURE": {"power backup", "parking", "generator", "lift",
                       "elevator", "water", "commercial", "road", "bore-well",
                       "borewell", "wi-fi", "wifi", "ac ", "pick and drop"},
}


def classify_amenity(text: str) -> str:
    lower = text.lower()
    for category, keywords in _AMENITY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "OTHER"


_LANDMARK_KEYWORDS: dict[str, list[str]] = {
    "EDUCATION":   ["school", "college", "vidyalaya", "university", "shala",
                    "institute", "academy"],
    "HEALTHCARE":  ["hospital", "clinic", "medical", "dispensary", "health"],
    "COMMERCIAL":  ["mall", "bazar", "bazaar", "republic", "market", "apmc",
                    "commercial", "shop"],
    "RELIGIOUS":   ["mandir", "temple", "masjid", "derasar", "church",
                    "mosque", "gurudwara", "upashray", "dehraser"],
    "TRANSPORT":   ["bus stop", "railway", "metro", "amts", "airport",
                    "ring road", "highway", "cross road", "circle", "chowkdi",
                    "over bridge", "flyover"],
    "RECREATION":  ["club", "park", "ground", "garden", "lake", "fun",
                    "recreation", "playground"],
}


def classify_landmark(name: str) -> str:
    lower = name.lower()
    for lm_type, keywords in _LANDMARK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return lm_type
    return "OTHER"


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING TEXT BUILDER  — combines ALL descriptive fields → one rich document
# ══════════════════════════════════════════════════════════════════════════════

def build_embedding_text(
    project: dict,
    unit: dict,
    unit_idx: int,
    amenities: list[str],
    landmarks: list[str],
) -> str:
    """
    Build a rich, production-quality text for ChromaDB embedding of one unit.

    Combines EVERY descriptive field:
      - Project identity (name, developer, location)
      - Society/project-level description
      - Unit description + specs (bhk, area, facing, applicable buildings)
      - Floor layout info (lifts, floors)
      - Room-level details (all rooms with names, types, dimensions, floor levels)
      - Amenities (full raw strings)
      - Nearby landmarks (all of them, with type classification)
      - RERA / status / date info

    This ensures a query like "4 BHK villa with home theatre near Karnavati Club"
    matches even if "home theatre" only appears in a room name or amenity string.
    """
    loc     = project.get("location") or {}
    society = project.get("society_layout") or {}
    rooms   = unit.get("rooms") or []
    parts: list[str] = []

    # ── Project identity ──────────────────────────────────────────────────────
    bhk       = _int(unit.get("bhk"))
    ptype     = _str(unit.get("property_type")) or "PROPERTY"
    city      = _str(loc.get("city")) or ""
    nbhd      = _str(loc.get("neighbourhood")) or ""
    proj_name = _str(project.get("project_name")) or ""
    developer = _str(project.get("developer_name")) or ""
    address   = _str(loc.get("address")) or ""
    unit_type = _str(unit.get("unit_type")) or ""
    facing    = _str(unit.get("entrance_Facing") or unit.get("entrance_facing"))

    parts.append(
        f"{bhk} BHK {ptype} in {nbhd}, {city}."
        f" Project: {proj_name} by {developer}."
    )

    if address:
        parts.append(f"Address: {address}.")

    # RERA / status
    rera   = _str(project.get("rera_registration_number"))
    status = _str(project.get("project_status"))
    pos_dt = _str(project.get("possession_date"))
    if rera:
        parts.append(f"RERA number: {rera}.")
    if status and status.upper() not in ("UNKNOWN", "NONE"):
        parts.append(f"Project status: {status}.")
    if pos_dt:
        parts.append(f"Possession date: {pos_dt}.")

    # Pin code
    pin = _str(loc.get("pin_code"))
    if pin:
        parts.append(f"Pin code: {pin}.")

    # ── Society description ───────────────────────────────────────────────────
    soc_desc = _str(society.get("description"))
    if soc_desc:
        parts.append(f"Society: {soc_desc}")

    # Society boolean flags (convert to readable text)
    soc_flags: list[str] = []
    if society.get("has_clubhouse"):        soc_flags.append("clubhouse")
    if society.get("has_park_or_garden"):   soc_flags.append("park or garden")
    if society.get("has_swimming_pool"):    soc_flags.append("swimming pool")
    if society.get("has_sports_courts"):    soc_flags.append("sports courts")
    if society.get("has_parking_area"):     soc_flags.append("parking")
    if society.get("commercial_shops_included"): soc_flags.append("commercial shops")
    if soc_flags:
        parts.append(f"Society facilities: {', '.join(soc_flags)}.")

    total_blocks = _int(society.get("total_apartment_blocks"))
    total_villas = _int(society.get("total_independent_villas_or_tenements"))
    bnames       = [b for b in (society.get("building_names") or []) if b]
    road_widths  = [r for r in (society.get("road_width_details") or []) if r]
    if total_blocks:
        parts.append(f"Total apartment blocks: {total_blocks}.")
    if total_villas:
        parts.append(f"Total villas/tenements: {total_villas}.")
    if bnames:
        parts.append(f"Buildings: {', '.join(bnames[:10])}.")  # cap at 10
    if road_widths:
        parts.append(f"Road widths: {', '.join(road_widths)}.")

    # ── Floor layouts (project-level) ──────────────────────────────────────────
    floor_layouts = project.get("floor_layouts") or []
    if floor_layouts:
        fl_descs: list[str] = []
        for fl in floor_layouts:
            fl_name  = _str(fl.get("layout_name")) or ""
            fl_units = _int(fl.get("total_units_on_floor"))
            fl_lift  = fl.get("has_lifts")
            fl_text  = fl_name
            if fl_units:
                fl_text += f" ({fl_units} units/floor)"
            if fl_lift:
                fl_text += ", with lifts"
            if fl_text.strip():
                fl_descs.append(fl_text)
        if fl_descs:
            parts.append(f"Floor layouts: {'; '.join(fl_descs)}.")

    # ── Unit description + specs ──────────────────────────────────────────────
    desc = _str(unit.get("description"))
    if desc:
        parts.append(desc)

    sba  = _float(unit.get("super_built_up_area_sqft"))
    ca   = _float(unit.get("carpet_area_sqft"))
    bal  = _float(unit.get("balcony_area_sqft"))
    wash = _float(unit.get("wash_area_sqft"))
    if sba:   parts.append(f"Super built-up area: {sba} sqft.")
    if ca:    parts.append(f"Carpet area: {ca} sqft.")
    if bal:   parts.append(f"Balcony area: {bal} sqft.")
    if wash:  parts.append(f"Wash area: {wash} sqft.")

    if unit_type and unit_type not in (f"{bhk} BHK", f"{bhk}BHK"):
        parts.append(f"Unit variant: {unit_type}.")
    if facing:
        parts.append(f"Entrance facing: {facing}.")

    appl_bldgs = [b for b in (unit.get("applicable_buildings") or []) if b]
    if appl_bldgs:
        parts.append(f"Applicable buildings: {', '.join(appl_bldgs)}.")

    # ── Rooms (ALL rooms with names, types, dimensions, floor levels) ─────────
    room_parts: list[str] = []
    for room in rooms:
        rname  = _str(room.get("name")) or ""
        rtype  = _str(room.get("room_type")) or ""
        rlen   = _str(room.get("length")) or ""
        rwid   = _str(room.get("width")) or ""
        rarea  = _float(room.get("area_sqft"))
        rfloor = _str(room.get("floor_level")) or ""
        r_ab   = room.get("attached_bathroom")
        r_bal  = room.get("has_balcony_access")

        room_str = rname
        if rtype and rtype.upper() not in ("OTHER",):
            room_str += f" ({rtype.replace('_', ' ').title()})"
        if rfloor:
            room_str += f" [{rfloor} floor]"
        dims: list[str] = []
        if rlen and rwid:
            dims.append(f"{rlen}×{rwid}")
        if rarea:
            dims.append(f"{rarea} sqft")
        if dims:
            room_str += f" {', '.join(dims)}"
        if r_ab:
            room_str += " [attached bath]"
        if r_bal:
            room_str += " [balcony access]"
        if room_str.strip():
            room_parts.append(room_str)

    if room_parts:
        parts.append(f"Rooms: {'; '.join(room_parts)}.")

    # ── Project amenities (full raw strings) ──────────────────────────────────
    if amenities:
        parts.append(f"Amenities: {'; '.join(amenities)}.")

    # ── Nearby landmarks (ALL of them + type) ─────────────────────────────────
    if landmarks:
        lm_with_types: list[str] = []
        for lm in landmarks:
            lm_type = classify_landmark(lm)
            lm_with_types.append(f"{lm} ({lm_type.title()})")
        parts.append(f"Nearby: {', '.join(lm_with_types)}.")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  CYPHER STATEMENTS
# ══════════════════════════════════════════════════════════════════════════════

# Project + City + Neighbourhood
_CYPHER_PROJECT = """
MERGE (city:City {name: $city})
MERGE (hood:Neighbourhood {name: $neighbourhood})
MERGE (hood)-[:IN_CITY]->(city)

MERGE (p:Project {project_id: $project_id})
SET p.project_name         = $project_name,
    p.developer_name       = $developer_name,
    p.rera_number          = $rera_number,
    p.project_status       = $project_status,
    p.possession_date      = $possession_date,
    p.address              = $address,
    p.pin_code             = $pin_code,
    p.has_clubhouse        = $has_clubhouse,
    p.has_pool             = $has_pool,
    p.has_park             = $has_park,
    p.has_sports_courts    = $has_sports_courts,
    p.has_parking          = $has_parking,
    p.has_commercial_shops = $has_commercial_shops,
    p.total_buildings      = $total_buildings,
    p.total_villas         = $total_villas,
    p.society_description  = $society_description,
    p.building_names       = $building_names,
    p.road_widths          = $road_widths,
    p.floor_layouts        = $floor_layouts,
    p.source_file          = $source_file,
    p.ingested_at          = $ingested_at

MERGE (p)-[:LOCATED_IN]->(hood)
MERGE (p)-[:IN_CITY]->(city)
"""

# Developer
_CYPHER_DEVELOPER = """
MERGE (dev:Developer {name: $dev_name})
MERGE (p:Project {project_id: $project_id})
MERGE (p)-[:BUILT_BY]->(dev)
"""

# Landmark
_CYPHER_LANDMARK = """
MERGE (lm:Landmark {name: $lm_name})
SET lm.landmark_type = $lm_type
MERGE (p:Project {project_id: $project_id})
MERGE (p)-[:NEAR]->(lm)
"""

# Amenity
_CYPHER_AMENITY = """
MERGE (am:Amenity {name: $am_name})
SET am.category = $am_category
MERGE (p:Project {project_id: $project_id})
MERGE (p)-[:HAS_AMENITY]->(am)
"""

# Unit  (MERGE so re-ingest is idempotent)
_CYPHER_UNIT = """
MERGE (u:Unit {unit_id: $unit_id})
SET u.project_id          = $project_id,
    u.unit_type           = $unit_type,
    u.property_type       = $property_type,
    u.bhk                 = $bhk,
    u.entrance_facing     = $entrance_facing,
    u.description         = $description,
    u.carpet_sqft         = $carpet_sqft,
    u.super_builtup_sqft  = $super_builtup_sqft,
    u.balcony_sqft        = $balcony_sqft,
    u.wash_sqft           = $wash_sqft,
    u.applicable_buildings= $applicable_buildings
MERGE (p:Project {project_id: $project_id})
MERGE (p)-[:HAS_UNIT]->(u)
"""

# Room  (CREATE — each room belongs to exactly one unit, never shared)
_CYPHER_ROOM = """
MATCH (u:Unit {unit_id: $unit_id})
CREATE (r:Room {
    room_type:          $room_type,
    name:               $name,
    length:             $length,
    width:              $width,
    area_sqft:          $area_sqft,
    floor_level:        $floor_level,
    attached_bathroom:  $attached_bathroom,
    has_balcony_access: $has_balcony_access
})
CREATE (u)-[:HAS_ROOM]->(r)
"""

# Delete all rooms for a unit (used on re-ingest with --force)
_CYPHER_DELETE_ROOMS = """
MATCH (u:Unit {unit_id: $unit_id})-[:HAS_ROOM]->(r:Room)
DETACH DELETE r
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SKIP-CHECK  — was this project already ingested?
# ══════════════════════════════════════════════════════════════════════════════

def is_already_ingested(driver, project_id: str) -> bool:
    with driver.session() as session:
        result = session.run(
            "MATCH (p:Project {project_id: $pid}) RETURN p.project_id LIMIT 1",
            pid=project_id,
        )
        return result.single() is not None


# ══════════════════════════════════════════════════════════════════════════════
#  CORE INGEST FUNCTION  — one JSON file → Neo4j + ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def ingest_one(
    json_file: Path,
    driver,
    chroma: chromadb.Collection,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    Ingest one brochure JSON into Neo4j + ChromaDB.

    Returns the number of units ingested (0 for skipped/dry-run).
    Raises on unrecoverable errors — caller logs and counts failures.
    """
    data: dict = json.loads(json_file.read_text(encoding="utf-8"))

    # ── Basic extraction ──────────────────────────────────────────────────────
    project_id   = normalise_project_id(
        data.get("project_id"), data.get("project_name", ""), json_file.name
    )
    project_name = _str(data.get("project_name")) or "Unknown Project"
    developer    = _str(data.get("developer_name")) or "Unknown Developer"
    loc          = data.get("location") or {}
    society      = data.get("society_layout") or {}
    amenities_raw: list[str] = [
        a for a in (data.get("amenities") or []) if _str(a)
    ]

    # ── Synthesise amenity strings from society_layout boolean flags ──────────
    # Some projects have has_clubhouse=true (etc.) but the amenities[] list
    # does not contain the matching text.  Without this step those projects
    # would have no HAS_AMENITY node for the feature, so amenity-based Cypher
    # filters would silently miss them.  We inject a canonical name only when
    # the list does not already contain the concept.
    _SOCIETY_FLAG_AMENITY: list[tuple[str, str]] = [
        ("has_clubhouse",            "Clubhouse"),
        ("has_swimming_pool",        "Swimming Pool"),
        ("has_park_or_garden",       "Garden / Park"),
        ("has_sports_courts",        "Sports Courts"),
        ("has_parking_area",         "Parking"),
        ("commercial_shops_included","Commercial Shops"),
    ]
    _amenities_lower = " ".join(amenities_raw).lower()
    for flag_key, canonical_name in _SOCIETY_FLAG_AMENITY:
        if society.get(flag_key):
            # Only add if no similar text is already in the list
            if canonical_name.lower().split("/")[0].strip() not in _amenities_lower:
                amenities_raw.append(canonical_name)
                logger.debug(
                    f"  ↪ Synthesised amenity '{canonical_name}' from flag "
                    f"'{flag_key}' for {data.get('project_name', '?')}"
                )

    landmarks_raw: list[str] = [
        lm for lm in (loc.get("nearby_landmarks") or []) if _str(lm)
    ]
    units: list[dict] = data.get("units") or []

    if dry_run:
        logger.info(
            f"  [DRY-RUN] {project_name} ({project_id}) — "
            f"{len(units)} unit(s), {len(landmarks_raw)} landmarks, "
            f"{len(amenities_raw)} amenities"
        )
        return 0

    # ── Already ingested? ─────────────────────────────────────────────────────
    if not force and is_already_ingested(driver, project_id):
        logger.debug(f"  ↷ Already ingested: {project_id}")
        return -1  # Caller treats -1 as "skipped"

    # ── Derived fields ─────────────────────────────────────────────────────────
    city         = normalise_city(loc.get("city"), loc.get("neighbourhood"))
    neighbourhood = _str(loc.get("neighbourhood")) or city or "Unknown"
    city          = city or neighbourhood

    floor_layouts_json = json.dumps(data.get("floor_layouts") or [])
    building_names_str = "|".join(
        b for b in (society.get("building_names") or []) if b
    )
    road_widths_str = " | ".join(
        r for r in (society.get("road_width_details") or []) if r
    )
    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with driver.session() as session:
        # ── 1. Project + City + Neighbourhood ───────────────────────────────
        session.run(_CYPHER_PROJECT, {
            "project_id":          project_id,
            "project_name":        project_name,
            "developer_name":      developer,
            "rera_number":         _str(data.get("rera_registration_number")),
            "project_status":      _str(data.get("project_status")),
            "possession_date":     _str(data.get("possession_date")),
            "city":                city,
            "neighbourhood":       neighbourhood,
            "address":             _str(loc.get("address")),
            "pin_code":            _str(loc.get("pin_code")),
            "has_clubhouse":       _bool_int(society.get("has_clubhouse")),
            "has_pool":            _bool_int(society.get("has_swimming_pool")),
            "has_park":            _bool_int(society.get("has_park_or_garden")),
            "has_sports_courts":   _bool_int(society.get("has_sports_courts")),
            "has_parking":         _bool_int(society.get("has_parking_area")),
            "has_commercial_shops":_bool_int(society.get("commercial_shops_included")),
            "total_buildings":     _int(society.get("total_apartment_blocks")),
            "total_villas":        _int(society.get("total_independent_villas_or_tenements")),
            "society_description": _str(society.get("description")),
            "building_names":      building_names_str,
            "road_widths":         road_widths_str,
            "floor_layouts":       floor_layouts_json,   # stored as JSON string
            "source_file":         json_file.name,
            "ingested_at":         ingested_at,
        })

        # ── 2. Developer ─────────────────────────────────────────────────────
        session.run(_CYPHER_DEVELOPER, {
            "dev_name":   developer,
            "project_id": project_id,
        })

        # ── 3. Landmarks ──────────────────────────────────────────────────────
        for lm_name in landmarks_raw:
            lm_clean = lm_name.strip()
            if not lm_clean:
                continue
            session.run(_CYPHER_LANDMARK, {
                "lm_name":   lm_clean,
                "lm_type":   classify_landmark(lm_clean),
                "project_id": project_id,
            })

        # ── 4. Amenities ──────────────────────────────────────────────────────
        for am_text in amenities_raw:
            am_clean = am_text.strip()
            if not am_clean:
                continue
            session.run(_CYPHER_AMENITY, {
                "am_name":     am_clean,
                "am_category": classify_amenity(am_clean),
                "project_id":  project_id,
            })

        # ── 5. Units + Rooms ─────────────────────────────────────────────────
        chroma_docs, chroma_metas, chroma_ids = [], [], []

        for idx, unit in enumerate(units):
            unit_id = f"{project_id}__{idx}"
            # Safely handle key inconsistency: entrance_Facing vs entrance_facing
            facing = _str(unit.get("entrance_Facing") or unit.get("entrance_facing"))

            appl = "|".join(
                b for b in (unit.get("applicable_buildings") or []) if b
            )

            # On force re-ingest, delete existing rooms for this unit first
            if force:
                session.run(_CYPHER_DELETE_ROOMS, {"unit_id": unit_id})

            session.run(_CYPHER_UNIT, {
                "unit_id":           unit_id,
                "project_id":        project_id,
                "unit_type":         _str(unit.get("unit_type")),
                "property_type":     _str(unit.get("property_type")),
                "bhk":               _int(unit.get("bhk")),
                "entrance_facing":   facing,
                "description":       _str(unit.get("description")),
                "carpet_sqft":       _float(unit.get("carpet_area_sqft")),
                "super_builtup_sqft":_float(unit.get("super_built_up_area_sqft")
                                           or unit.get("super_builtup_sqft")),
                "balcony_sqft":      _float(unit.get("balcony_area_sqft")),
                "wash_sqft":         _float(unit.get("wash_area_sqft")),
                "applicable_buildings": appl,
            })

            # Rooms — CREATE (not MERGE)
            for room in (unit.get("rooms") or []):
                session.run(_CYPHER_ROOM, {
                    "unit_id":          unit_id,
                    "room_type":        _str(room.get("room_type")),
                    "name":             _str(room.get("name")),
                    "length":           _str(room.get("length")),
                    "width":            _str(room.get("width")),
                    "area_sqft":        _float(room.get("area_sqft")),
                    "floor_level":      _str(room.get("floor_level")),
                    "attached_bathroom":_bool_int(room.get("attached_bathroom")),
                    "has_balcony_access":_bool_int(room.get("has_balcony_access")),
                })

            # ChromaDB document
            text = build_embedding_text(
                project=data,
                unit=unit,
                unit_idx=idx,
                amenities=amenities_raw,
                landmarks=landmarks_raw,
            )
            chroma_docs.append(text)
            chroma_ids.append(unit_id)
            chroma_metas.append({
                "project_id":    project_id,
                "unit_id":       unit_id,
                "project_name":  project_name,
                "city":          city or "",
                "neighbourhood": neighbourhood or "",
                "bhk":           _int(unit.get("bhk")) or 0,
                "property_type": (_str(unit.get("property_type")) or "").upper(),
                "area_sqft":     float(_float(unit.get("super_built_up_area_sqft")) or 0),
                "developer":     _str(data.get("developer_name")),
                "amenities":     ", ".join(amenities_raw)[:800],
            })

        # ── 6. ChromaDB batch upsert ──────────────────────────────────────────
        batch_size = settings.INGEST_BATCH_SIZE
        for i in range(0, len(chroma_docs), batch_size):
            chroma.upsert(
                documents=chroma_docs[i : i + batch_size],
                metadatas=chroma_metas[i : i + batch_size],
                ids=chroma_ids[i : i + batch_size],
            )

    return len(units)


# ══════════════════════════════════════════════════════════════════════════════
#  CHROMADB SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.CHROMA_EMBED_MODEL
    )
    col = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' "
        f"ready (existing docs: {col.count()})"
    )
    return col


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(
    json_dir: Path,
    workers: int,
    force: bool,
    dry_run: bool,
) -> None:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}")
        return

    total = len(json_files)
    logger.info(f"Found {total} JSON file(s) in '{json_dir}'")
    logger.info(f"Neo4j: {settings.NEO4J_URI} | Workers: {workers} | Force: {force}")

    # ── Connect ───────────────────────────────────────────────────────────────
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    logger.success("Connected to Neo4j.")

    chroma = get_chroma_collection()

    # ── Optional wipe ─────────────────────────────────────────────────────────
    if force and not dry_run:
        logger.warning("--force: wiping existing graph and ChromaDB collection ...")
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        # Reset ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        chroma_client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        chroma = get_chroma_collection()
        logger.warning("Graph and ChromaDB wiped.")

    # ── Run parallel ingestion ─────────────────────────────────────────────────
    ingested = skipped = failed = 0
    total_units = 0
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(ingest_one, f, driver, chroma, dry_run, force): f
            for f in json_files
        }

        completed = 0
        for future in as_completed(futures):
            f = futures[future]
            completed += 1
            try:
                result = future.result()
                if result == -1:
                    skipped += 1
                    logger.debug(f"  ↷ Skipped: {f.name}")
                elif result >= 0:
                    total_units += result
                    ingested += 1
                    logger.success(f"  ✓ {f.name} ({result} units)")
            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Failed: {f.name} → {e}")

            # Progress log every 100 files
            if completed % 100 == 0:
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = total - completed
                eta = remaining / rate if rate > 0 else 0
                logger.info(
                    f"  ⏱  Progress: {completed}/{total} files "
                    f"| {rate:.1f} files/sec "
                    f"| ETA: {eta:.0f}s"
                )

    elapsed_total = time.perf_counter() - t_start
    driver.close()

    sep = "─" * 60
    logger.info(
        f"\n{sep}\n"
        f"  Ingestion complete{'  [DRY-RUN]' if dry_run else ''}:\n"
        f"  ✓ {ingested} new  |  ↷ {skipped} skipped  |  ✗ {failed} failed\n"
        f"  Total units stored: {total_units}\n"
        f"  Elapsed: {elapsed_total:.1f}s\n"
        f"  Neo4j   → {settings.NEO4J_URI}\n"
        f"  ChromaDB → {settings.CHROMA_PATH}\n"
        f"{sep}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/kg_ingest_{time:YYYYMMDD_HHmmss}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )

    parser = argparse.ArgumentParser(
        description="Ingest brochure JSONs into Neo4j KG + ChromaDB"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the entire graph + ChromaDB collection and re-ingest all files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all JSONs and print what would be ingested — no writes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.INGEST_WORKERS,
        help=f"Parallel thread count (default: {settings.INGEST_WORKERS})",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=settings.JSON_OUTPUT_DIR,
        help=f"JSON directory (default: {settings.JSON_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    run_ingestion(
        json_dir=args.dir,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
    )
