"""
ingest.py — Batch ingestion from brochure JSONs into:
  1. SQLite   → projects, project_landmarks, project_amenities, units, rooms tables
  2. ChromaDB → one embedding per unit (description + context)

Features:
  - Idempotent: skips already-ingested project_ids
  - Batch ChromaDB upserts (INGEST_BATCH_SIZE at a time)
  - Handles all JSON field inconsistencies (entrance_Facing vs entrance_facing, etc.)
  - Computes all boolean flags and key room metrics during ingestion
  - Rebuilds FTS5 index after each project

Usage:
    python ingest.py              # ingest new files only
    python ingest.py --force      # re-ingest everything (clears DB first)
    python ingest.py --dry-run    # validate JSONs without writing to DB
"""

import argparse
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from logger import logger
from schema import create_all_tables, get_connection


# ─── ChromaDB Setup ───────────────────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.CHROMA_EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' ready ({collection.count()} docs).")
    return collection


# ─── Room Analysis ────────────────────────────────────────────────────────────

ROOM_TYPE_FLAGS = {
    "POOJA_ROOM":   "has_pooja_room",
    "STUDY_ROOM":   "has_study_room",
    "TERRACE":      "has_terrace",
    "SERVANT_ROOM": "has_servant_room",
    "DRESSING_ROOM":"has_dressing_room",
    "STORE_ROOM":   "has_store_room",
    "COURTYARD":    "has_courtyard",
    "LOBBY":        "has_lobby",
    "BALCONY":      "has_balcony",
}

NAME_KEYWORD_FLAGS = {
    "has_garden":       {"garden", "courtyard"},
    "has_home_theatre": {"theatre", "theater", "home theater"},
    "has_gym":          {"gym", "gymnasium", "home gym"},
}


def analyze_rooms(rooms: list) -> dict:
    """Extract all derived metrics from a unit's room list."""
    flags = {v: 0 for v in ROOM_TYPE_FLAGS.values()}
    flags.update({"has_garden": 0, "has_home_theatre": 0, "has_gym": 0})

    total_rooms = len(rooms)
    total_bedrooms = 0
    total_toilets = 0
    attached_bathrooms = 0
    bedroom_areas = []
    drawing_room_sqft = None
    kitchen_sqft = None

    for room in rooms:
        rt = (room.get("room_type") or "").upper()
        name_lower = (room.get("name") or "").lower()
        area = room.get("area_sqft")

        # ── room_type flags ──
        if rt in ROOM_TYPE_FLAGS:
            flags[ROOM_TYPE_FLAGS[rt]] = 1

        # ── name keyword flags ──
        for flag, keywords in NAME_KEYWORD_FLAGS.items():
            if any(kw in name_lower for kw in keywords):
                flags[flag] = 1

        # ── special cases: HOME THEATRE / HOME GYM in name with OTHER type ──
        if rt == "OTHER":
            if any(kw in name_lower for kw in NAME_KEYWORD_FLAGS["has_home_theatre"]):
                flags["has_home_theatre"] = 1
            if any(kw in name_lower for kw in NAME_KEYWORD_FLAGS["has_gym"]):
                flags["has_gym"] = 1

        # ── room type counters ──
        if rt == "BEDROOM":
            total_bedrooms += 1
            if area:
                bedroom_areas.append(area)
            if room.get("attached_bathroom"):
                attached_bathrooms += 1

        elif rt in ("TOILET", "BATHROOM", "WC"):
            total_toilets += 1

        elif rt == "DRAWING_ROOM" and area and drawing_room_sqft is None:
            drawing_room_sqft = area

        elif rt == "KITCHEN" and area and kitchen_sqft is None:
            kitchen_sqft = area

    master_bedroom_sqft = max(bedroom_areas) if bedroom_areas else None

    return {
        **flags,
        "total_rooms":        total_rooms,
        "total_bedrooms":     total_bedrooms,
        "total_toilets":      total_toilets,
        "attached_bathrooms": attached_bathrooms,
        "master_bedroom_sqft":master_bedroom_sqft,
        "drawing_room_sqft":  drawing_room_sqft,
        "kitchen_sqft":       kitchen_sqft,
    }


# ─── Embedding Text Builder ───────────────────────────────────────────────────

def build_unit_embedding_text(project: dict, unit: dict, amenities: list, landmarks: list) -> str:
    """Build a rich, descriptive text for ChromaDB embedding of a unit."""
    loc = project.get("location", {})
    society = project.get("society_layout", {}) or {}

    parts = []

    # Identity
    bhk = unit.get("bhk")
    ptype = unit.get("property_type", "")
    city = loc.get("city", "")
    neighbourhood = loc.get("neighbourhood", "")
    proj_name = project.get("project_name", "")
    developer = project.get("developer_name", "")

    parts.append(f"{bhk} BHK {ptype} in {neighbourhood}, {city} by {developer} ({proj_name}).")

    # Unit description
    desc = unit.get("description") or ""
    if desc:
        parts.append(desc)

    # Areas
    sba = unit.get("super_built_up_area_sqft") or unit.get("super_builtup_sqft")
    ca = unit.get("carpet_area_sqft")
    if sba:
        parts.append(f"Super built-up area: {sba} sqft.")
    if ca:
        parts.append(f"Carpet area: {ca} sqft.")

    # Room highlights (special rooms only — not toilets/kitchens)
    rooms = unit.get("rooms", [])
    special_rooms = [
        r.get("name") for r in rooms
        if r.get("room_type") in (
            "POOJA_ROOM", "STUDY_ROOM", "TERRACE", "SERVANT_ROOM",
            "DRESSING_ROOM", "STORE_ROOM", "COURTYARD", "BALCONY",
        ) or any(kw in (r.get("name") or "").lower()
                 for kw in ("garden","theatre","theater","gym","gymnasium"))
    ]
    if special_rooms:
        parts.append(f"Special rooms: {', '.join(r for r in special_rooms if r)}.")

    # Unit type (villa variant name etc.)
    unit_type = unit.get("unit_type", "")
    if unit_type and unit_type not in (f"{bhk} BHK",):
        parts.append(f"Unit variant: {unit_type}.")

    # Entrance facing
    facing = unit.get("entrance_Facing") or unit.get("entrance_facing")
    if facing:
        parts.append(f"Entrance facing: {facing}.")

    # Project amenities
    if amenities:
        parts.append(f"Project amenities: {'; '.join(amenities)}.")

    # Society description
    soc_desc = society.get("description", "")
    if soc_desc:
        parts.append(soc_desc)

    # Nearby landmarks (most important ones, first 8)
    if landmarks:
        parts.append(f"Nearby: {', '.join(landmarks[:8])}.")

    return " ".join(parts)


# ─── Project Ingestion ────────────────────────────────────────────────────────

_NULL_STRINGS = {"null", "none", "na", "n/a", "", "undefined"}


def _normalise_project_id(raw_id, project_name: str, source_file: str) -> str:
    """
    Return a stable, unique project_id.
    - If the raw_id from JSON is a real value (not null/empty), use it as-is.
    - Otherwise build a slug from project_name + source_file so every brochure
      gets its own deterministic ID (no two null-id brochures collide).
    """
    if raw_id and str(raw_id).strip().lower() not in _NULL_STRINGS:
        return str(raw_id).strip()
    # Deterministic fallback: slug-ify project name + stem of source filename
    import re as _re
    name_part = _re.sub(r"[^\w]+", "_", (project_name or "unknown").strip())[:40]
    file_part = _re.sub(r"[^\w]+", "_", Path(source_file).stem)[:20]
    return f"{name_part}__{file_part}"


def is_ingested(conn: sqlite3.Connection, project_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)).fetchone()
    return row is not None


# ─── City Normalisation ───────────────────────────────────────────────────────
# Brochure LLMs sometimes return city names in Gujarati, ALL-CAPS, or use a
# neighbourhood name as the city. Normalise everything to consistent English
# title-case so searches like "list all projects in Ahmedabad" never miss data.

# Gujarati city names → English
_GUJARATI_CITY_MAP: dict[str, str] = {
    "અમદાવાદ": "Ahmedabad",
    "સુરત":    "Surat",
    "વડોદરા":  "Vadodara",
    "રાજકોટ":  "Rajkot",
    "ભાવનગર": "Bhavnagar",
    "ગાંધીનગર":"Gandhinagar",
    "જામનગર": "Jamnagar",
    "જૂનાગઢ":  "Junagadh",
    "આણંદ":   "Anand",
    "નડિયાદ":  "Nadiad",
    "મહેસાણા": "Mehsana",
    "પોરબંદર": "Porbandar",
    "મોરબી":  "Morbi",
    "ભૂજ":    "Bhuj",
    "વલસાડ":  "Valsad",
    "ગાંઘ":   "Gandhinagar",
}

# Known Ahmedabad sub-areas sometimes stored as the city name
_AHMEDABAD_LOCALITIES: set[str] = {
    "isanpur", "nikol", "new nikol", "naroda", "new naroda",
    "bopal", "new bopal", "sarkhej", "thaltej", "prahlad nagar",
    "satellite", "bodakdev", "vastrapur", "maninagar", "ghatlodia",
    "chandkheda", "ranip", "vastral road", "vastrapur road",
    "science city road", "sg highway", "sg road", "shela",
    "shantigram", "south bopal", "ambli", "shilaj",
    "vejalpur", "vinzol", "vizol", "hanspur", "hanspura", "gamdi",
    "chiloda", "kotarpur", "chiloda-kotarpur",
}

# Known Surat sub-areas
_SURAT_LOCALITIES: set[str] = {
    "adajan", "pal", "sarthana", "katargam", "varachha",
    "vesu", "piplod", "jahangirpura",
}

_LOCALITY_TO_CITY: dict[str, str] = {}
for _loc in _AHMEDABAD_LOCALITIES:
    _LOCALITY_TO_CITY[_loc] = "Ahmedabad"
for _loc in _SURAT_LOCALITIES:
    _LOCALITY_TO_CITY[_loc] = "Surat"


def _normalise_city(raw_city: str | None, neighbourhood: str | None = None) -> str | None:
    """
    Return a normalised English city name.
    Handles: Gujarati script, ALL-CAPS, whitespace, and neighbourhood-as-city.
    Falls back to neighbourhood→city lookup, then returns None if genuinely unknown.
    """
    # 1. Gujarati → English
    if raw_city and raw_city.strip() in _GUJARATI_CITY_MAP:
        return _GUJARATI_CITY_MAP[raw_city.strip()]

    # 2. Strip + title-case normalise
    if raw_city:
        city_clean = raw_city.strip()
        # Title-case if all-caps (AHMEDABAD → Ahmedabad)
        if city_clean.isupper():
            city_clean = city_clean.title()
        # Check if the "city" is actually a locality name
        if city_clean.lower() in _LOCALITY_TO_CITY:
            return _LOCALITY_TO_CITY[city_clean.lower()]
        return city_clean

    # 3. city is None/empty — try to infer from neighbourhood
    if neighbourhood:
        nbhd_lower = neighbourhood.strip().lower()
        if nbhd_lower in _LOCALITY_TO_CITY:
            return _LOCALITY_TO_CITY[nbhd_lower]

    return None  # genuinely unknown


def ingest_project(
    conn: sqlite3.Connection,
    collection: chromadb.Collection,
    data: dict,
    source_file: str,
    dry_run: bool = False,
) -> int:
    """Ingest one project. Returns number of units ingested."""

    raw_pid = data.get("project_id")
    project_id = _normalise_project_id(raw_pid, data.get("project_name", ""), source_file)
    loc = data.get("location", {}) or {}
    society = data.get("society_layout", {}) or {}
    amenities_list = data.get("amenities", []) or []
    landmarks_list = (loc.get("nearby_landmarks") or [])

    if dry_run:
        logger.info(f"  [DRY-RUN] Would ingest: {data.get('project_name')} ({len(data.get('units', []))} units)")
        return len(data.get("units", []))

    now = datetime.now(timezone.utc).isoformat()

    # Normalise city at ingestion time for consistent search
    raw_city = loc.get("city")
    neighbourhood = loc.get("neighbourhood")
    normalised_city = _normalise_city(raw_city, neighbourhood)
    if raw_city and normalised_city != raw_city:
        logger.debug(f"  City normalised: {repr(raw_city)} → {repr(normalised_city)}")

    # ── Insert project ──
    conn.execute(
        """INSERT OR REPLACE INTO projects VALUES (
            :project_id, :project_name, :developer_name, :rera_number, :project_status,
            :possession_date, :city, :neighbourhood, :address, :pin_code,
            :has_clubhouse, :has_pool, :has_park, :has_sports_courts, :has_parking,
            :has_commercial_shops, :total_buildings, :total_villas, :society_description,
            :road_widths, :source_file, :ingested_at
        )""",
        {
            "project_id":          project_id,
            "project_name":        data.get("project_name"),
            "developer_name":      data.get("developer_name"),
            "rera_number":         data.get("rera_registration_number"),
            "project_status":      data.get("project_status"),
            "possession_date":     data.get("possession_date"),
            "city":                normalised_city,

            "neighbourhood":       loc.get("neighbourhood"),
            "address":             loc.get("address"),
            "pin_code":            loc.get("pin_code"),
            "has_clubhouse":       int(bool(society.get("has_clubhouse"))),
            "has_pool":            int(bool(society.get("has_swimming_pool"))),
            "has_park":            int(bool(society.get("has_park_or_garden"))),
            "has_sports_courts":   int(bool(society.get("has_sports_courts"))),
            "has_parking":         int(bool(society.get("has_parking_area"))),
            "has_commercial_shops":int(bool(society.get("commercial_shops_included"))),
            "total_buildings":     society.get("total_apartment_blocks"),
            "total_villas":        society.get("total_independent_villas_or_tenements"),
            "society_description": society.get("description"),
            "road_widths":         " | ".join(society.get("road_width_details") or []),
            "source_file":         source_file,
            "ingested_at":         now,
        },
    )

    # Remove existing child rows (for re-ingestion)
    conn.execute("DELETE FROM project_landmarks WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM project_amenities WHERE project_id = ?", (project_id,))

    # ── Insert landmarks ──
    for landmark in landmarks_list:
        if landmark and landmark.strip():
            conn.execute(
                "INSERT INTO project_landmarks (project_id, landmark_name) VALUES (?, ?)",
                (project_id, landmark.strip()),
            )

    # ── Insert amenities ──
    for amenity in amenities_list:
        if amenity and amenity.strip():
            conn.execute(
                "INSERT INTO project_amenities (project_id, amenity) VALUES (?, ?)",
                (project_id, amenity.strip()),
            )

    # ── Update FTS5 index ──
    landmarks_text = " ".join(landmarks_list)
    amenities_text = " ".join(amenities_list)

    conn.execute("DELETE FROM projects_fts WHERE project_id = ?", (project_id,))
    conn.execute(
        """INSERT INTO projects_fts (project_id, project_name, developer_name,
           city, neighbourhood, address, society_description, landmarks_text, amenities_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            project_id,
            data.get("project_name", ""),
            data.get("developer_name", ""),
            normalised_city or "",
            loc.get("neighbourhood", ""),
            loc.get("address", ""),
            society.get("description", ""),
            landmarks_text,
            amenities_text,
        ),
    )

    # ── Insert units + rooms ──
    units = data.get("units", []) or []
    chroma_docs, chroma_meta, chroma_ids = [], [], []

    for idx, unit in enumerate(units):
        # Normalize entrance_facing key inconsistency from gemini.py schema
        facing = unit.get("entrance_Facing") or unit.get("entrance_facing")

        unit_id = f"{project_id}__{idx}"
        unit_id_safe = unit_id.replace(" ", "_").replace("/", "-")

        rooms = unit.get("rooms", []) or []
        metrics = analyze_rooms(rooms)

        # Remove existing unit rows (for re-ingestion)
        conn.execute("DELETE FROM rooms WHERE unit_id = ?", (unit_id_safe,))
        conn.execute("DELETE FROM units WHERE unit_id = ?", (unit_id_safe,))

        conn.execute(
            """INSERT INTO units VALUES (
                :unit_id, :project_id, :unit_type, :property_type, :bhk, :entrance_facing,
                :description, :carpet_area_sqft, :super_builtup_sqft, :balcony_area_sqft,
                :wash_area_sqft, :applicable_buildings,
                :total_rooms, :total_bedrooms, :total_toilets, :attached_bathrooms,
                :master_bedroom_sqft, :drawing_room_sqft, :kitchen_sqft,
                :has_pooja_room, :has_study_room, :has_terrace, :has_servant_room,
                :has_garden, :has_home_theatre, :has_gym, :has_dressing_room,
                :has_store_room, :has_courtyard, :has_lobby, :has_balcony
            )""",
            {
                "unit_id":            unit_id_safe,
                "project_id":         project_id,
                "unit_type":          unit.get("unit_type"),
                "property_type":      unit.get("property_type"),
                "bhk":                unit.get("bhk"),
                "entrance_facing":    facing,
                "description":        unit.get("description"),
                "carpet_area_sqft":   unit.get("carpet_area_sqft"),
                "super_builtup_sqft": unit.get("super_built_up_area_sqft"),
                "balcony_area_sqft":  unit.get("balcony_area_sqft"),
                "wash_area_sqft":     unit.get("wash_area_sqft"),
                "applicable_buildings": ",".join(unit.get("applicable_buildings") or []),
                **metrics,
            },
        )

        # ── Insert rooms ──
        for room in rooms:
            conn.execute(
                """INSERT INTO rooms (unit_id, project_id, name, room_type, length, width,
                   area_sqft, floor_level, attached_bathroom, has_balcony_access)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    unit_id_safe,
                    project_id,
                    room.get("name"),
                    room.get("room_type"),
                    room.get("length"),
                    room.get("width"),
                    room.get("area_sqft"),
                    room.get("floor_level"),
                    int(bool(room.get("attached_bathroom"))) if room.get("attached_bathroom") is not None else None,
                    int(bool(room.get("has_balcony_access"))) if room.get("has_balcony_access") is not None else None,
                ),
            )

        # ── Build ChromaDB text ──
        text = build_unit_embedding_text(data, unit, amenities_list, landmarks_list)
        chroma_docs.append(text)
        chroma_ids.append(unit_id_safe)
        chroma_meta.append({
            "project_id":    project_id,
            "unit_id":       unit_id_safe,
            "project_name":  data.get("project_name", ""),
            "city":          normalised_city or "",
            "bhk":           unit.get("bhk") or 0,
            "property_type": (unit.get("property_type") or "").upper(),
            "area_sqft":     float(unit.get("super_built_up_area_sqft") or 0),
        })

    # ── Batch upsert to ChromaDB ──
    batch_size = settings.INGEST_BATCH_SIZE
    for i in range(0, len(chroma_docs), batch_size):
        collection.upsert(
            documents=chroma_docs[i : i + batch_size],
            metadatas=chroma_meta[i : i + batch_size],
            ids=chroma_ids[i : i + batch_size],
        )

    conn.commit()
    return len(units)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_ingestion(force: bool = False, dry_run: bool = False) -> None:
    json_dir = settings.JSON_OUTPUT_DIR
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}.")
        return

    logger.info(f"Found {len(json_files)} JSON file(s) in '{json_dir}'")

    conn = get_connection()
    create_all_tables(conn)

    if force and not dry_run:
        logger.warning("--force: clearing existing data.")
        conn.executescript("""
            DELETE FROM rooms;
            DELETE FROM units;
            DELETE FROM project_landmarks;
            DELETE FROM project_amenities;
            DELETE FROM projects;
            DELETE FROM projects_fts;
        """)
        conn.commit()

    collection = get_chroma_collection()

    ingested, skipped, failed = 0, 0, 0
    total_units = 0
    t_start = time.perf_counter()

    for i, json_file in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] {json_file.name}")
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            project_id = _normalise_project_id(
                data.get("project_id"), data.get("project_name", ""), json_file.name
            )

            if not force and is_ingested(conn, project_id):
                logger.info(f"  ↳ Already ingested ({project_id}) — skipping.")
                skipped += 1
                continue

            n = ingest_project(conn, collection, data, json_file.name, dry_run=dry_run)
            total_units += n
            ingested += 1
            logger.success(f"  ✓ {data.get('project_name')} — {n} unit(s)")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed += 1

        # Batch progress logging every 100 files
        if i % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(f"  ⏱  Progress: {i}/{len(json_files)} files ({rate:.1f} files/sec)")

    elapsed_total = time.perf_counter() - t_start
    conn.close()
    separator = "─" * 55
    logger.info(
        f"\n{separator}\n"
        f"  Ingestion complete{'  [DRY-RUN]' if dry_run else ''}:\n"
        f"  ✓ {ingested} new  |  ↷ {skipped} skipped  |  ✗ {failed} failed\n"
        f"  Total units stored: {total_units}\n"
        f"  Elapsed: {elapsed_total:.1f}s\n"
        f"  SQLite  → {settings.sqlite_path}\n"
        f"  ChromaDB → {settings.chroma_path}\n"
        f"{separator}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest brochure JSONs into the RERA database.")
    parser.add_argument("--force",   action="store_true", help="Re-ingest all files (clears DB first)")
    parser.add_argument("--dry-run", action="store_true", help="Validate JSONs without writing to DB")
    args = parser.parse_args()

    run_ingestion(force=args.force, dry_run=args.dry_run)
