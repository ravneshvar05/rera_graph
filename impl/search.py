"""
search.py — 4-step search engine:
  1. SQL structured filter  (SQLite: city, bhk, property_type, feature flags)
  2. FTS5 text search       (landmarks, amenities, project name full-text)
  3. ChromaDB vector search (semantic similarity on unit descriptions)
  4. Merge + score + group  (group by project, return top-K unique projects)

For AGGREGATE queries: skips vectors, runs SQL COUNT/GROUP BY directly.
For DETAIL queries: fetches full project with all rooms.
"""

import sqlite3
import time
from typing import Any, Optional
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from config import settings
from logger import logger
from query_planner import SearchPlan


# ─── Connections (lazy singletons) ───────────────────────────────────────────

_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA foreign_keys=ON;")
    return _conn


@st.cache_resource(show_spinner="Loading AI search model...")
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.CHROMA_EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Step 1: SQL Structured Filter ───────────────────────────────────────────

UNIT_FLAG_COLUMNS = {
    "has_pooja_room", "has_study_room", "has_terrace", "has_servant_room",
    "has_garden", "has_home_theatre", "has_gym", "has_dressing_room",
    "has_store_room", "has_courtyard", "has_lobby", "has_balcony",
}

PROJECT_FLAG_COLUMNS = {
    "has_clubhouse", "has_pool", "has_park", "has_sports_courts", "has_parking",
}

# Property types that are considered "non-apartment" for fallback logic
NON_APARTMENT_TYPES = {"VILLA", "ROW_HOUSE", "TENEMENT", "PENTHOUSE"}


def _build_sql_filter(plan: SearchPlan) -> tuple[str, list]:
    conditions, params = [], []

    # City filter — supports multiple cities
    if plan.cities:
        placeholders = ",".join("?" * len(plan.cities))
        conditions.append(f"LOWER(p.city) IN ({placeholders})")
        params.extend(c.lower() for c in plan.cities)

    # BHK: exact match OR minimum ("3 BHK or larger") OR specific list ("3 or 5 BHK")
    if plan.bhk is not None:
        conditions.append("u.bhk = ?")
        params.append(plan.bhk)
    elif getattr(plan, "min_bhk", None) is not None:
        conditions.append("u.bhk >= ?")
        params.append(plan.min_bhk)
    elif getattr(plan, "bhk_options", None):
        placeholders = ",".join("?" * len(plan.bhk_options))
        conditions.append(f"u.bhk IN ({placeholders})")
        params.extend(plan.bhk_options)

    if plan.property_type:
        conditions.append("u.property_type = ?")
        params.append(plan.property_type.upper())

    if plan.min_area_sqft:
        conditions.append("(u.super_builtup_sqft >= ? OR u.carpet_area_sqft >= ?)")
        params.extend([plan.min_area_sqft, plan.min_area_sqft])

    if plan.max_area_sqft:
        conditions.append("(u.super_builtup_sqft <= ? OR u.carpet_area_sqft <= ?)")
        params.extend([plan.max_area_sqft, plan.max_area_sqft])

    if plan.entrance_facing:
        conditions.append("LOWER(u.entrance_facing) = ?")
        params.append(plan.entrance_facing.lower())

    # Unit feature flags
    for flag in (plan.must_have or []):
        if flag in UNIT_FLAG_COLUMNS:
            conditions.append(f"u.{flag} = 1")

    # Project feature flags
    for flag in (plan.project_must_have or []):
        if flag in PROJECT_FLAG_COLUMNS:
            conditions.append(f"p.{flag} = 1")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


def sql_search(plan: SearchPlan, limit: int = 300) -> list[dict]:
    """Run structured SQL search, returns unit+project dicts."""
    where, params = _build_sql_filter(plan)

    sql = f"""
        SELECT
            u.unit_id, u.unit_type, u.property_type, u.bhk, u.entrance_facing,
            u.description, u.carpet_area_sqft, u.super_builtup_sqft,
            u.balcony_area_sqft, u.wash_area_sqft, u.applicable_buildings,
            u.total_rooms, u.total_bedrooms, u.total_toilets, u.attached_bathrooms,
            u.master_bedroom_sqft, u.drawing_room_sqft, u.kitchen_sqft,
            u.has_pooja_room, u.has_study_room, u.has_terrace, u.has_servant_room,
            u.has_garden, u.has_home_theatre, u.has_gym, u.has_dressing_room,
            u.has_store_room, u.has_courtyard, u.has_lobby, u.has_balcony,
            p.project_id, p.project_name, p.developer_name, p.rera_number,
            p.project_status, p.possession_date,
            p.city, p.neighbourhood, p.address,
            p.has_clubhouse, p.has_pool, p.has_park, p.has_sports_courts,
            p.has_parking, p.has_commercial_shops,
            p.total_buildings, p.total_villas, p.society_description
        FROM units u
        JOIN projects p ON u.project_id = p.project_id
        {where}
        ORDER BY
            CASE WHEN p.city IS NOT NULL THEN 0 ELSE 1 END,
            CASE WHEN u.bhk IS NOT NULL THEN 0 ELSE 1 END,
            u.bhk ASC
        LIMIT {limit}
    """
    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    result = [dict(r) for r in rows]
    logger.debug(f"SQL returned {len(result)} row(s)")
    return result


# ─── Location Variant Expansion ──────────────────────────────────────────────
# Some areas have "New" prefix variants or known alternate spellings that users
# omit. This maps a user term to extra DB search terms to try alongside it.

_LOCATION_VARIANTS: dict[str, list[str]] = {
    # Ahmedabad east areas
    "nikol":        ["nikol", "new nikol"],
    "new nikol":    ["new nikol", "nikol"],
    "naroda":       ["naroda", "new naroda"],
    "new naroda":   ["new naroda", "naroda"],
    "bopal":        ["bopal", "new bopal"],
    "ranip":        ["ranip", "new ranip"],
    # Common FTS spelling variants
    "vizol":        ["vizol", "vinzol"],
    "vinzol":       ["vinzol", "vizol"],
    "vejalpur":     ["vejalpur", "vejalpore"],
}


def _expand_locations(locations: list[str]) -> list[str]:
    """Expand location list with known variants (e.g. Nikol → [Nikol, New Nikol])."""
    expanded: list[str] = []
    seen: set[str] = set()
    for loc in locations:
        variants = _LOCATION_VARIANTS.get(loc.lower(), [loc])
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                expanded.append(v)
    return expanded


# ─── Step 2: FTS5 Text Search (Tiered) ───────────────────────────────────────

def fts5_search(
    plan: SearchPlan,
) -> tuple[set[str], set[str]]:
    """
    Tiered location search. Returns two sets of project_ids:

      strong_ids — project neighbourhood or address directly matches the location
                   (i.e. the project IS in that area)
      weak_ids   — project has the location word only in landmarks or amenities
                   (i.e. the project is NEAR something called that)

    Additional amenity/project_name filters are applied via FTS5 on top.

    Logic:
    1. SQL LIKE on neighbourhood + address → strong_ids  (authoritative)
    2. FTS5 full-text on all columns → all_fts_ids
    3. weak_ids = all_fts_ids - strong_ids
    4. If locations given but strong_ids empty, weak_ids acts as fallback.
    """
    has_locations = bool(getattr(plan, "locations", None))
    has_amenity = bool(getattr(plan, "amenity_query", None))
    has_proj_name = bool(getattr(plan, "project_name_query", None))

    if not has_locations and not has_amenity and not has_proj_name:
        return set(), set()

    conn = _get_conn()
    strong_ids: set[str] = set()
    all_fts_ids: set[str] = set()

    # ── Expand location variants (Nikol → New Nikol, Vizol → Vinzol, etc.) ──
    raw_locations = list(getattr(plan, "locations", []) or [])
    expanded_locations = _expand_locations(raw_locations) if raw_locations else []

    # ── Step A: SQL LIKE on neighbourhood + address (STRONG match) ──────────
    if expanded_locations:
        try:
            like_conds, like_params = [], []
            for loc in expanded_locations:
                pattern = f"%{loc}%"
                like_conds.append(
                    "(LOWER(p.neighbourhood) LIKE LOWER(?) OR LOWER(p.address) LIKE LOWER(?))"
                )
                like_params.extend([pattern, pattern])
            rows = conn.execute(
                f"SELECT DISTINCT p.project_id FROM projects p WHERE {' OR '.join(like_conds)}",
                like_params,
            ).fetchall()
            strong_ids = {r["project_id"] for r in rows}
            if strong_ids:
                logger.debug(f"Strong (address) match: {len(strong_ids)} project(s) for {expanded_locations}")
        except Exception as e:
            logger.warning(f"SQL LIKE (strong) failed: {e}")

    # ── Step B: FTS5 full-text search (covers landmarks + amenities too) ─────
    fts_parts: list[str] = []

    if raw_locations:
        loc_and_groups = []
        for loc in raw_locations:
            variants = _LOCATION_VARIANTS.get(loc.lower(), [loc])
            or_terms = []
            for v in variants:
                safe = str(v).replace('"', '')
                if " " in safe.strip():
                    or_terms.append(f'"{safe}"')
                else:
                    or_terms.append(safe)
            loc_and_groups.append(f"({' OR '.join(or_terms)})")
        fts_parts.append(f"({' AND '.join(loc_and_groups)})")

    if has_amenity:
        aq = plan.amenity_query
        if isinstance(aq, list):
            aq = " ".join(str(x) for x in aq)
        safe_amenity = str(aq).replace('"', '""')
        fts_parts.append(f'"{safe_amenity}"')

    if has_proj_name:
        pnq = plan.project_name_query
        if isinstance(pnq, list):
            pnq = " ".join(str(x) for x in pnq)
        safe_proj = str(pnq).replace('"', '""')
        fts_parts.append(f'"{safe_proj}"')

    if fts_parts:
        match_str = " AND ".join(fts_parts)
        try:
            rows = conn.execute(
                'SELECT project_id FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank',
                (match_str,),
            ).fetchall()
            all_fts_ids = {r["project_id"] for r in rows}
            logger.debug(f"FTS5 matched {len(all_fts_ids)} project(s) for: {match_str}")
        except Exception as e:
            logger.warning(f"FTS5 query '{match_str}' failed: {e}")

    # ── Derive weak_ids: FTS hit but NOT a direct address/neighbourhood match ─
    weak_ids = all_fts_ids - strong_ids

    # ── For amenity-only / project-name-only queries (no location) ────────────
    # Promote everything to strong so the downstream filter works as before.
    if not has_locations:
        strong_ids = all_fts_ids
        weak_ids = set()

    logger.debug(
        f"fts5_search → strong={len(strong_ids)} weak={len(weak_ids)} "
        f"(locs={raw_locations} expanded={expanded_locations})"
    )
    return strong_ids, weak_ids


# ─── Step 3: ChromaDB Vector Search ──────────────────────────────────────────

def vector_search(plan: SearchPlan, top_k: int = 50) -> list[tuple[str, float]]:
    """
    Run semantic search in ChromaDB.
    Returns list of (unit_id, similarity_score) sorted by best match.
    """
    if not plan.semantic_query:
        return []

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    # Build metadata pre-filter (narrows search space)
    filters = []
    if len(plan.cities) == 1:
        filters.append({"city": {"$eq": plan.cities[0]}})
    if plan.bhk is not None:
        filters.append({"bhk": {"$eq": plan.bhk}})
    if plan.property_type:
        filters.append({"property_type": {"$eq": plan.property_type.upper()}})

    kwargs: dict[str, Any] = {
        "query_texts": [plan.semantic_query],
        "n_results": min(top_k, count),
        "include": ["distances", "metadatas"],
    }
    if len(filters) == 1:
        kwargs["where"] = filters[0]
    elif len(filters) > 1:
        kwargs["where"] = {"$and": filters}

    try:
        results = collection.query(**kwargs)
        ids = results["ids"][0]
        distances = results["distances"][0]   # cosine distance: lower = better
        scores = [(uid, 1.0 - dist) for uid, dist in zip(ids, distances)]
        logger.debug(f"ChromaDB returned {len(scores)} result(s)")
        return scores
    except Exception as e:
        logger.warning(f"ChromaDB search failed: {e}")
        return []


# ─── Step 4: Merge, Score, and Group by Project ───────────────────────────────

def merge_and_group(
    sql_results: list[dict],
    fts5_project_ids: set[str],
    vector_scores: list[tuple[str, float]],
    plan: SearchPlan,
    top_k: int = 5,
    strong_project_ids: set[str] | None = None,
    weak_project_ids: set[str] | None = None,
) -> list[dict]:
    """
    Merge SQL, FTS5, and vector results with a weighted score.
    Group by project, return top_k unique projects.

    Scoring:
      - SQL match:    base score 0.4  (unit passed hard filters)
      - FTS5 hit:     bonus 0.3       (project has landmark/amenity match)
      - Vector score: weighted 0.5 × similarity (best semantic match)
    """
    sql_map = {r["unit_id"]: r for r in sql_results}
    vector_map = {uid: score for uid, score in vector_scores}

    # Collect all candidate unit_ids
    all_unit_ids: set[str] = set(sql_map.keys()) | set(vector_map.keys())

    scored_units: list[tuple[float, dict]] = []

    has_locations = bool(getattr(plan, "locations", None))
    has_amenity = bool(getattr(plan, "amenity_query", None))
    has_proj_name = bool(getattr(plan, "project_name_query", None))
    requires_fts5 = has_locations or has_amenity or has_proj_name

    # ── Tiered location logic ────────────────────────────────────────────────
    # strong_project_ids: project neighbourhood/address directly matches location
    # weak_project_ids:   project has location word only in landmarks/amenities
    # fts5_project_ids:   union of strong + weak (backwards compat for non-location filters)
    strong_ids = strong_project_ids if strong_project_ids is not None else fts5_project_ids
    weak_ids = weak_project_ids if weak_project_ids is not None else set()
    all_matching_ids = strong_ids | weak_ids | fts5_project_ids

    # If we have strong hits for a location query, ONLY show strong matches.
    # If no strong hits but weak hits exist, use weak (landmark proximity) — caller
    # will annotate the response as approximate.
    if has_locations and strong_ids:
        permitted_ids = strong_ids
    elif has_locations and weak_ids:
        permitted_ids = weak_ids
    elif requires_fts5:
        permitted_ids = all_matching_ids
    else:
        permitted_ids = None  # No location filter — all SQL results allowed

    has_any_fts_hits = bool(all_matching_ids)

    for uid in all_unit_ids:
        unit = sql_map.get(uid)
        if unit is None:
            # Vector hit not in SQL — skip (doesn't pass hard filters)
            continue

        pid = unit.get("project_id")

        # If location/amenity filter requested AND we have hits, enforce the filter.
        # If FTS returned 0 hits entirely (area not in DB or name not found), NO items are permitted.
        if requires_fts5:
            if not has_any_fts_hits:
                continue  # Nothing matched the FTS requirement
            if permitted_ids is not None and pid not in permitted_ids:
                continue

        score = 0.4  # base for SQL match

        # Score bonuses by tier
        if pid in strong_ids:
            score += 0.4   # strong: project IS in that area
        elif pid in weak_ids:
            score += 0.15  # weak: landmark proximity only
        elif pid in fts5_project_ids:
            score += 0.3   # amenity/project-name FTS hit

        # Vector similarity
        if uid in vector_map:
            score += 0.5 * vector_map[uid]

        # FIX: Only apply the 0.55 threshold when vector search ran
        # (i.e., when we have actual similarity scores to compare).
        # For SQL-only results (no vectors), use base threshold of 0.4.
        min_score = 0.55 if vector_scores else 0.4
        if score < min_score:
            continue

        scored_units.append((score, unit))

    # Within each project, select best-scoring unit
    project_best_score: dict[str, float] = {}
    project_units: dict[str, list[dict]] = {}

    for score, unit in scored_units:
        pid = unit["project_id"]
        if pid not in project_best_score or score > project_best_score[pid]:
            project_best_score[pid] = score
        project_units.setdefault(pid, []).append(unit)

    # Sort projects by best-unit score descending
    sorted_projects = sorted(project_best_score.keys(), key=lambda p: -project_best_score[p])

    results = []
    for pid in sorted_projects[:top_k]:
        units_in_project = project_units[pid]
        # Use any unit for project-level fields
        sample = units_in_project[0]

        # Get landmarks and amenities
        conn = _get_conn()
        landmarks = [r[0] for r in conn.execute(
            "SELECT landmark_name FROM project_landmarks WHERE project_id = ? ORDER BY id",
            (pid,),
        ).fetchall()]
        amenities = [r[0] for r in conn.execute(
            "SELECT amenity FROM project_amenities WHERE project_id = ? ORDER BY id",
            (pid,),
        ).fetchall()]

        proj = {
            "project_id":        pid,
            "project_name":      sample["project_name"],
            "developer_name":    sample["developer_name"],
            "city":              sample["city"],
            "neighbourhood":     sample["neighbourhood"],
            "address":           sample.get("address"),
            "project_status":    sample.get("project_status"),
            "possession_date":   sample.get("possession_date"),
            "has_clubhouse":     sample["has_clubhouse"],
            "has_pool":          sample["has_pool"],
            "has_park":          sample["has_park"],
            "has_sports_courts": sample["has_sports_courts"],
            "has_parking":       sample["has_parking"],
            "total_buildings":   sample.get("total_buildings"),
            "total_villas":      sample.get("total_villas"),
            "society_description": sample.get("society_description"),
            "nearby_landmarks":  landmarks,
            "amenities":         amenities,
            "relevance_score":   round(project_best_score[pid], 3),
            "matching_units":    units_in_project,
        }
        results.append(proj)

    logger.info(f"Merged: {len(results)} unique project(s) (from {len(sql_map)} SQL + {len(vector_map)} vector)")
    return results


# ─── Aggregate Query Handler ──────────────────────────────────────────────────

def aggregate_search(plan: SearchPlan) -> dict:
    """
    Handle AGGREGATE queries without vector search.
    Returns a summary dict with counts AND matching project cards.
    """
    conn = _get_conn()
    where, params = _build_sql_filter(plan)

    base_sql = f"""
        FROM units u
        JOIN projects p ON u.project_id = p.project_id
        {where}
    """

    # Total unit count
    total = conn.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()[0]

    # Projects by city
    city_rows = conn.execute(
        f"SELECT p.city, COUNT(DISTINCT p.project_id) as cnt {base_sql} GROUP BY p.city ORDER BY cnt DESC",
        params,
    ).fetchall()

    # BHK breakdown
    bhk_rows = conn.execute(
        f"SELECT u.bhk, COUNT(*) as cnt {base_sql} GROUP BY u.bhk ORDER BY u.bhk",
        params,
    ).fetchall()

    # Property type breakdown
    type_rows = conn.execute(
        f"SELECT u.property_type, COUNT(*) as cnt {base_sql} GROUP BY u.property_type ORDER BY cnt DESC",
        params,
    ).fetchall()

    # ── Also fetch matching project cards so the UI can render them ──
    unit_rows = sql_search(plan, limit=200)
    project_cards = merge_and_group(
        sql_results=unit_rows,
        fts5_project_ids=set(),
        vector_scores=[],
        plan=plan,
        top_k=10,   # show up to 10 matching projects for aggregates
    )

    return {
        "total_units":      total,
        "by_city":          [dict(r) for r in city_rows],
        "by_bhk":           [dict(r) for r in bhk_rows],
        "by_property_type": [dict(r) for r in type_rows],
        "project_cards":    project_cards,
    }


# ─── Detail Query Handler ──────────────────────────────────────────────────────

def detail_search(plan: SearchPlan) -> list[dict]:
    """
    Fetch full project details including all rooms for DETAIL queries.
    """
    conn = _get_conn()

    # Find project by name using FTS first, then fallback to LIKE
    strong_ids, weak_ids = fts5_search(plan)
    project_ids = list(strong_ids | weak_ids)

    if not project_ids and plan.project_name_query:
        rows = conn.execute(
            "SELECT project_id FROM projects WHERE LOWER(project_name) LIKE ?",
            (f"%{plan.project_name_query.lower()}%",),
        ).fetchall()
        project_ids = [r["project_id"] for r in rows]

    if not project_ids:
        return []

    results = []
    for pid in project_ids[:2]:  # max 2 projects for detail
        proj_row = conn.execute("SELECT * FROM projects WHERE project_id = ?", (pid,)).fetchone()
        if not proj_row:
            continue
        proj = dict(proj_row)

        units = conn.execute("SELECT * FROM units WHERE project_id = ?", (pid,)).fetchall()
        proj["matching_units"] = []

        for u in units:
            unit = dict(u)
            rooms = conn.execute(
                "SELECT * FROM rooms WHERE unit_id = ? ORDER BY id", (u["unit_id"],)
            ).fetchall()
            unit["rooms"] = [dict(r) for r in rooms]
            proj["matching_units"].append(unit)

        proj["nearby_landmarks"] = [r[0] for r in conn.execute(
            "SELECT landmark_name FROM project_landmarks WHERE project_id = ?", (pid,)
        ).fetchall()]
        proj["amenities"] = [r[0] for r in conn.execute(
            "SELECT amenity FROM project_amenities WHERE project_id = ?", (pid,)
        ).fetchall()]

        proj["relevance_score"] = 1.0   # detail queries are always 100% relevant
        results.append(proj)

    return results


# ─── Compare Query Handler ────────────────────────────────────────────────────

def compare_search(plan: SearchPlan) -> list[dict]:
    """
    Handle COMPARE queries by finding each named project individually.
    Uses FTS for each project name, falling back to LIKE search.
    Returns a list of project dicts (one per named project).
    """
    conn = _get_conn()

    # Build the list of project names to look up
    names_to_find: list[str] = list(plan.compare_projects) if plan.compare_projects else []
    if not names_to_find and plan.project_name_query:
        names_to_find = [plan.project_name_query]

    if not names_to_find:
        return []

    results = []
    found_pids: set[str] = set()

    for name in names_to_find:
        pid: Optional[str] = None

        # Try FTS match
        try:
            safe = name.replace('"', '""')
            rows = conn.execute(
                'SELECT project_id FROM projects_fts WHERE projects_fts MATCH ? ORDER BY rank LIMIT 1',
                (f'"{safe}"',),
            ).fetchall()
            if rows:
                pid = rows[0]["project_id"]
        except Exception as e:
            logger.warning(f"FTS compare search failed for '{name}': {e}")

        # Fallback: LIKE on project_name
        if not pid:
            row = conn.execute(
                "SELECT project_id FROM projects WHERE LOWER(project_name) LIKE ? LIMIT 1",
                (f"%{name.lower()}%",),
            ).fetchone()
            if row:
                pid = row["project_id"]

        if not pid or pid in found_pids:
            continue
        found_pids.add(pid)

        proj_row = conn.execute("SELECT * FROM projects WHERE project_id = ?", (pid,)).fetchone()
        if not proj_row:
            continue
        proj = dict(proj_row)

        units = conn.execute("SELECT * FROM units WHERE project_id = ?", (pid,)).fetchall()
        proj["matching_units"] = [dict(u) for u in units]

        proj["nearby_landmarks"] = [r[0] for r in conn.execute(
            "SELECT landmark_name FROM project_landmarks WHERE project_id = ?", (pid,)
        ).fetchall()]
        proj["amenities"] = [r[0] for r in conn.execute(
            "SELECT amenity FROM project_amenities WHERE project_id = ?", (pid,)
        ).fetchall()]

        proj["relevance_score"] = 1.0
        results.append(proj)

    logger.info(f"compare_search: found {len(results)} project(s) for {names_to_find}")
    return results


# ─── Non-Apartment Fallback Helper ───────────────────────────────────────────

def _search_non_apartment(plan: SearchPlan, top_k: int = 5) -> list[dict]:
    """
    Fallback search when a specific non-apartment type returns 0 results.
    Broadens to ALL non-apartment types (VILLA, ROW_HOUSE, TENEMENT, PENTHOUSE)
    while keeping city, BHK, location and other filters intact.
    If still 0 results, drops the BHK filter as well to show available inventory.
    """
    import copy
    fallback_plan = copy.copy(plan)
    fallback_plan.property_type = None  # Remove the tight type filter

    sql_results = sql_search(fallback_plan, limit=settings.MAX_SQL_CANDIDATES)

    # Keep only non-apartment rows
    sql_results = [r for r in sql_results if r.get("property_type", "").upper() != "APARTMENT"]

    # ── FIX: If still 0 results, drop the BHK filter too (e.g. they asked for 3 BHK VILLA but we only have 4 BHK VILLA)
    if not sql_results and fallback_plan.bhk is not None:
        logger.info(f"Fallback 2: No non-apartments with {fallback_plan.bhk} BHK. Dropping BHK filter.")
        fallback_plan2 = copy.copy(fallback_plan)
        fallback_plan2.bhk = None
        sql_results = sql_search(fallback_plan2, limit=settings.MAX_SQL_CANDIDATES)
        sql_results = [r for r in sql_results if r.get("property_type", "").upper() != "APARTMENT"]
        fallback_plan = fallback_plan2  # use this plan for FTS and Vector

    if not sql_results:
        return []

    strong_ids, weak_ids = fts5_search(fallback_plan)
    fts5_hits = strong_ids | weak_ids
    vector_scores = vector_search(fallback_plan, top_k=settings.MAX_VECTOR_RESULTS)

    return merge_and_group(
        sql_results, fts5_hits, vector_scores, fallback_plan, top_k=top_k,
        strong_project_ids=strong_ids, weak_project_ids=weak_ids,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def search(plan: SearchPlan, top_k: int = 5) -> dict:
    """
    Main search function. Routes based on query_type.

    Returns:
        {
          "query_type": "SEARCH" | "AGGREGATE" | "COMPARE" | "DETAIL",
          "results": [...]          # list of project dicts for SEARCH/COMPARE/DETAIL
          "aggregate": {...}        # summary dict for AGGREGATE
          "fallback_non_apt": bool  # True if results are a non-apt fallback
          "original_type": str      # the originally requested type (e.g. "TENEMENT")
          "location_tier": str|None # "exact" | "landmark_proximity" | None
          "searched_locations": list# locations the user asked for
        }
    """
    searched_locations = list(getattr(plan, "locations", []) or [])

    if plan.needs_clarification:
        return {"query_type": "SEARCH", "results": [], "aggregate": None,
                "fallback_non_apt": False, "original_type": None,
                "location_tier": None, "searched_locations": searched_locations}

    qt = plan.query_type.upper()

    if qt == "AGGREGATE":
        agg = aggregate_search(plan)
        project_cards = agg.pop("project_cards", [])
        return {"query_type": "AGGREGATE", "results": project_cards, "aggregate": agg,
                "fallback_non_apt": False, "original_type": None,
                "location_tier": None, "searched_locations": searched_locations}

    if qt == "DETAIL":
        results = detail_search(plan)
        return {"query_type": "DETAIL", "results": results, "aggregate": None,
                "fallback_non_apt": False, "original_type": None,
                "location_tier": None, "searched_locations": searched_locations}

    if qt == "COMPARE":
        results = compare_search(plan)
        return {"query_type": "COMPARE", "results": results, "aggregate": None,
                "fallback_non_apt": False, "original_type": None,
                "location_tier": None, "searched_locations": searched_locations}

    # ── SEARCH — full pipeline with performance timing ─────────────────────────
    t0 = time.perf_counter()

    sql_results = sql_search(plan, limit=settings.MAX_SQL_CANDIDATES)
    t1 = time.perf_counter()
    logger.info(f"⏱  SQL search: {t1-t0:.3f}s, {len(sql_results)} row(s)")

    if not sql_results:
        logger.info("No SQL results — falling back to pure semantic search.")

    strong_ids, weak_ids = fts5_search(plan)
    fts5_hits = strong_ids | weak_ids
    t2 = time.perf_counter()
    logger.info(f"⏱  FTS5 search: {t2-t1:.3f}s, strong={len(strong_ids)} weak={len(weak_ids)}")

    vector_scores = vector_search(plan, top_k=settings.MAX_VECTOR_RESULTS)
    t3 = time.perf_counter()
    logger.info(f"⏱  Vector search: {t3-t2:.3f}s, {len(vector_scores)} result(s)")

    # For pure semantic fallback (no SQL results), include vector-only hits
    if not sql_results and vector_scores:
        conn_tmp = _get_conn()
        for uid, _ in vector_scores[:20]:
            row = conn_tmp.execute(
                """SELECT u.*, p.project_id as proj_id, p.project_name, p.developer_name,
                   p.city, p.neighbourhood, p.address, p.project_status, p.possession_date,
                   p.has_clubhouse, p.has_pool, p.has_park, p.has_sports_courts,
                   p.has_parking, p.has_commercial_shops, p.total_buildings,
                   p.total_villas, p.society_description
                   FROM units u JOIN projects p ON u.project_id = p.project_id
                   WHERE u.unit_id = ?""",
                (uid,),
            ).fetchone()
            if row:
                sql_results.append(dict(row))

    results = merge_and_group(
        sql_results, fts5_hits, vector_scores, plan, top_k=top_k,
        strong_project_ids=strong_ids, weak_project_ids=weak_ids,
    )

    # Determine location tier for honest messaging in the answer generator
    has_locations = bool(searched_locations)
    if not has_locations:
        location_tier = None
    elif strong_ids and any(p["project_id"] in strong_ids for p in results):
        location_tier = "exact"
    elif weak_ids and results:
        location_tier = "landmark_proximity"
    else:
        location_tier = None  # no location results at all
    t4 = time.perf_counter()
    logger.info(f"⏱  Total search pipeline: {t4-t0:.3f}s")

    # ── Non-apartment fallback ────────────────────────────────────────────────
    # If the user asked for a specific non-apartment type but got 0 results,
    # broaden to ALL non-apartment types so we never return an empty hand.
    original_type = plan.property_type.upper() if plan.property_type else None
    if not results and original_type and original_type in NON_APARTMENT_TYPES:
        logger.info(
            f"No results for {original_type} — broadening to all non-apartment types."
        )
        results = _search_non_apartment(plan, top_k=top_k)
        if results:
            return {
                "query_type": qt,
                "results": results,
                "aggregate": None,
                "fallback_non_apt": True,
                "original_type": original_type,
                "location_tier": location_tier,
                "searched_locations": searched_locations,
            }

    return {
        "query_type": qt,
        "results": results,
        "aggregate": None,
        "fallback_non_apt": False,
        "original_type": original_type,
        "location_tier": location_tier,
        "searched_locations": searched_locations,
    }
