"""
graphrag_retriever.py — Dual retrieval from Neo4j (graph) + ChromaDB (vector).

Given a QueryIntent, this module:
  1. Runs dynamic Cypher queries against Neo4j to find structurally matching projects
  2. Runs semantic similarity search in ChromaDB for fuzzy/subjective queries
  3. Merges and deduplicates results into a unified context string for the LLM

The merged context is the "retrieved knowledge" passed to graphrag_answer.py.

Performance: Graph and Vector retrieval run in PARALLEL via ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from neo4j import GraphDatabase

from graphrag_config import settings
from graphrag_intent import QueryIntent
from graphrag_cypher import generate_cypher, CypherQuery


# ── In-process query result cache ─────────────────────────────────────────────
# Stores (timestamp, result_tuple) keyed by normalised query string.
# Avoids re-running the full pipeline (LLM + Neo4j + ChromaDB) for repeated
# identical queries within the same Streamlit session (e.g. sidebar buttons).
_QUERY_CACHE: dict[str, tuple[float, tuple]] = {}
_CACHE_TTL_SECONDS: int = 300   # 5 minutes
_CACHE_MAX_SIZE: int = 50       # max entries before oldest is evicted



# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ProjectResult:
    """Unified result from either graph or vector retrieval."""
    project_id:    str
    project_name:  str
    city:          str
    neighbourhood: str
    developer:     str
    units:         list[dict]      = field(default_factory=list)
    floor_layouts: list[dict]      = field(default_factory=list)
    amenities:     list[str]       = field(default_factory=list)
    landmarks:     list[str]       = field(default_factory=list)
    extra_props:   dict            = field(default_factory=dict)
    score:         float           = 1.0   # vector similarity score (lower = more similar)
    source:        str             = "graph"  # "graph", "vector", or "both"

    def to_context_text(self) -> str:
        """Convert to a rich text block for LLM context assembly."""
        lines = [
            f"PROJECT: {self.project_name}",
            f"  Developer: {self.developer}",
            f"  Location: {self.neighbourhood}, {self.city}",
        ]
        if self.extra_props.get("address"):
            lines.append(f"  Address: {self.extra_props['address']}")
        if self.extra_props.get("project_status"):
            lines.append(f"  Status: {self.extra_props['project_status']}")
        if self.extra_props.get("possession_date"):
            lines.append(f"  Possession: {self.extra_props['possession_date']}")
        if self.extra_props.get("rera_number"):
            lines.append(f"  RERA: {self.extra_props['rera_number']}")

        # Society features
        soc = []
        if self.extra_props.get("has_clubhouse") == 1:   soc.append("Clubhouse")
        if self.extra_props.get("has_pool") == 1:         soc.append("Swimming Pool")
        if self.extra_props.get("has_park") == 1:         soc.append("Park/Garden")
        if self.extra_props.get("has_parking") == 1:      soc.append("Parking")
        if self.extra_props.get("has_sports_courts") == 1: soc.append("Sports Courts")
        if soc:
            lines.append(f"  Society Features: {', '.join(soc)}")

        if self.extra_props.get("society_description"):
            lines.append(f"  Society Description: {self.extra_props['society_description']}")

        # Units
        if self.units:
            lines.append("  Units:")
            for u in self.units:
                u_line = f"    - {u.get('unit_type', '?')} ({u.get('property_type', '?')})"
                if u.get("super_builtup_sqft"):
                    u_line += f" {u['super_builtup_sqft']} sqft super built-up"
                if u.get("carpet_sqft"):
                    u_line += f" / {u['carpet_sqft']} sqft carpet"
                if u.get("balcony_sqft"):
                    u_line += f" | Balcony: {u['balcony_sqft']} sqft"
                if u.get("entrance_facing"):
                    u_line += f" | Facing: {u['entrance_facing']}"
                if u.get("description"):
                    u_line += f"\n      Description: {u['description']}"
                lines.append(u_line)
                
                # Append Rooms if available
                if u.get("rooms"):
                    rooms_list = []
                    for r in u["rooms"]:
                        r_desc = r.get("name", "Room")
                        dims = []
                        if r.get("length") and r.get("width"):
                            dims.append(f"{r['length']} x {r['width']}")
                        if r.get("area_sqft"):
                            dims.append(f"{r['area_sqft']} sqft")
                        if dims:
                            r_desc += f" ({', '.join(dims)})"
                        if r.get("attached_bathroom") == 1:
                            r_desc += " with attached bath"
                        rooms_list.append(r_desc)
                    if rooms_list:
                        lines.append(f"      Rooms: {'; '.join(rooms_list)}")

        # Floor Layouts
        if getattr(self, "floor_layouts", None):
            lines.append("  Floor Layouts:")
            for f in self.floor_layouts:
                f_line = f"    - {f.get('layout_name', 'Unnamed Layout')}: {f.get('total_units_on_floor', 'Unknown')} units per floor"
                if f.get("has_lifts") == 1:
                    f_line += ", Lifts available"
                lines.append(f_line)

        # Amenities
        if self.amenities:
            lines.append(f"  Amenities: {', '.join(self.amenities)}")

        # Landmarks
        if self.landmarks:
            lines.append(f"  Nearby: {', '.join(self.landmarks[:10])}")

        # All other extra properties not explicitly formatted above
        already_used = {
            "address", "project_status", "possession_date", "rera_number",
            "has_clubhouse", "has_pool", "has_park", "has_parking", "has_sports_courts",
            "society_description"
        }
        other_props = []
        for k, v in self.extra_props.items():
            if k not in already_used and v is not None and v != "" and v != []:
                # format keys from snake_case to readable text
                readable_k = k.replace("_", " ").title()
                other_props.append(f"{readable_k}: {v}")
        if other_props:
            lines.append(f"  Other Details: {', '.join(other_props)}")

        lines.append(f"  [Source: {self.source}]")
        return "\n".join(lines)


# ── Amenity keyword → Project boolean flag mapping ────────────────────────────
# When a user asks for an amenity that maps to a direct boolean property on the
# Project node, we can also check that flag in Cypher (in addition to the
# HAS_AMENITY relationship).  This means a project is retrieved even if its
# amenity *text* doesn't mention the feature but the structured field does.
_AMENITY_TO_FLAG: dict[str, str] = {
    "clubhouse":       "p.has_clubhouse",
    "club house":      "p.has_clubhouse",
    "club":            "p.has_clubhouse",
    "pool":            "p.has_pool",
    "swimming pool":   "p.has_pool",
    "swimming":        "p.has_pool",
    "park":            "p.has_park",
    "garden":          "p.has_park",
    "lawn":            "p.has_park",
    "sports":          "p.has_sports_courts",
    "sports court":    "p.has_sports_courts",
    "badminton":       "p.has_sports_courts",
    "tennis":          "p.has_sports_courts",
    "volleyball":      "p.has_sports_courts",
    "parking":         "p.has_parking",
    "car park":        "p.has_parking",
    "commercial":      "p.has_commercial_shops",
    "shop":            "p.has_commercial_shops",
}


def _amenity_flag(amenity_text: str) -> str | None:
    """Return the Neo4j Project property name that corresponds to this amenity
    keyword, or None if no direct mapping exists."""
    lower = amenity_text.lower().strip()
    # Exact match first
    if lower in _AMENITY_TO_FLAG:
        return _AMENITY_TO_FLAG[lower]
    # Substring match (e.g. 'swimming pool facility' → 'swimming pool')
    for keyword, flag in _AMENITY_TO_FLAG.items():
        if keyword in lower:
            return flag
    return None


# ── Neo4j / Graph Retriever ────────────────────────────────────────────────────

class GraphRetriever:
    """Retrieves projects from Neo4j using dynamic Cypher queries."""


    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.debug("GraphRetriever connected to Neo4j.")

    def close(self):
        self._driver.close()

    def retrieve(self, cypher_result: CypherQuery) -> tuple[list[ProjectResult], list[dict]]:
        """
        Execute a Text-to-Cypher generated query against Neo4j.

        Strategy:
          - GLOBAL → return all projects.
          - SPECIFIC → run LLM Cypher, then ALWAYS supplement with intent-driven
            results for multi-location queries (LLM often drops locations).
            Also fallback to intent-driven on 0 results, BUT ONLY if the intent
            has at least one actionable graph filter (location, BHK, amenity, etc.).
            If intent has NO filters, the LLM Cypher 0-result means "nothing in db
            matches this structural constraint" — skip fallback so vector search
            can answer the semantic query instead.
          - LOOKUP/AGGREGATE → run LLM Cypher only (project name based).

        Returns:
            (project_results, answer_data)
            - answer_data is non-empty only for LOOKUP / AGGREGATE queries
        """
        if cypher_result.query_type == "GLOBAL":
            results = self._get_all_projects()
            answer_data: list[dict] = []
        else:
            results, answer_data = self._execute_cypher(cypher_result)

            # ── Intent-driven supplement / fallback for SPECIFIC queries ──
            if cypher_result.query_type == "SPECIFIC" and cypher_result.intent:
                intent = cypher_result.intent
                need_supplement = False

                # Case 1: LLM Cypher returned 0 results — fallback ONLY if the
                # intent has at least one graph-expressible filter.
                # Example of "no graph filters": "ground floor villas for seniors"
                # → property_type=VILLA but no location/BHK/amenity → LLM filter
                # (floor_level=0) fails but intent-fallback would dump ALL villas,
                # which is too broad. Let vector search handle it instead.
                if not results:
                    if self._intent_has_graph_filters(intent):
                        need_supplement = True
                        logger.info(
                            "LLM Cypher returned 0 graph results. "
                            "Intent has graph filters — using intent-driven fallback."
                        )
                    else:
                        logger.info(
                            "LLM Cypher returned 0 graph results. "
                            "Intent has no actionable graph filters — skipping fallback, "
                            "relying on vector search."
                        )

                # Case 2: Multi-location query — LLM often drops some
                # locations from Cypher params, so supplement to catch them
                elif isinstance(intent.neighbourhood, list) and len(intent.neighbourhood) > 1:
                    need_supplement = True
                    logger.info(
                        f"Multi-location query ({len(intent.neighbourhood)} locations). "
                        "Supplementing LLM Cypher with intent-driven results."
                    )

                if need_supplement:
                    fallback_results = self._get_filtered_projects(intent)
                    if fallback_results:
                        # Merge by project_id — avoid duplicates
                        existing_ids = {r.project_id for r in results}
                        added = 0
                        for fr in fallback_results:
                            if fr.project_id not in existing_ids:
                                results.append(fr)
                                existing_ids.add(fr.project_id)
                                added += 1
                        if added:
                            logger.info(f"Intent-driven supplement added {added} new project(s)")

        logger.info(f"Graph retrieval → {len(results)} project(s)")
        return results, answer_data

    @staticmethod
    def _intent_has_graph_filters(intent: QueryIntent) -> bool:
        """
        Returns True if the intent has at least one filter that can be
        expressed as a structural graph query (location, BHK, amenity, etc.).

        Used to gate the intent-driven fallback: if the LLM Cypher returned
        0 results but the intent has NO graph-expressible filters, we skip
        the fallback so vector search can handle the semantic query instead
        of flooding the context with all projects.

        Filters that DON'T count (can't be expressed as simple graph filters):
          - semantic_keywords only  (e.g. "spacious", "senior-friendly")
          - property_type alone  (too broad — would return all villas/apartments)
          - has_balcony / has_parking alone  (already covered by graph flags)
        """
        if intent.city:
            return True
        if intent.neighbourhood:
            return True
        if intent.zone:
            return True
        if intent.bhk is not None:
            return True
        if intent.amenities:
            return True
        if intent.landmark_types:
            return True
        if intent.specific_landmarks:
            return True
        if intent.developer:
            return True
        if intent.project_names:
            return True
        if intent.min_sqft is not None or intent.max_sqft is not None:
            return True
        if intent.min_units_per_floor is not None or intent.max_units_per_floor is not None:
            return True
        if intent.entrance_facing:
            return True
        # property_type alone is intentionally NOT listed — too broad to be useful
        # as a sole fallback filter (would return all apartments or all villas)
        return False

    def _execute_cypher(self, cq: CypherQuery) -> tuple[list[ProjectResult], list[dict]]:
        """Execute the LLM-generated Cypher and return (project_results, answer_data)."""
        try:
            with self._driver.session() as session:
                records = list(session.run(cq.cypher, **cq.params))

                # AGGREGATE/LOOKUP queries may only return `answer_data` (no `p` node).
                # Only call _record_to_result for rows that actually have a `p` key.
                results: list[ProjectResult] = []
                for r in records:
                    try:
                        if r.get("p") is not None:
                            results.append(self._record_to_result(r))
                    except (KeyError, TypeError):
                        pass  # row has no `p` node — it's an answer_data-only row

                # Extract answer_data from LOOKUP / AGGREGATE queries
                answer_data: list[dict] = []
                if cq.answer_columns:
                    for record in records:
                        for col in cq.answer_columns:
                            val = record.get(col)
                            if val:
                                # val is a list of dicts (one per unit) or a single dict
                                if isinstance(val, list):
                                    answer_data.extend([dict(v) for v in val if v])
                                elif isinstance(val, dict):
                                    answer_data.append(dict(val))

                return results, answer_data
        except Exception as e:
            logger.warning(f"Text-to-Cypher execution failed: {e}")
            logger.warning(f"Failed Cypher:\n{cq.cypher}")
            logger.warning(f"Params: {cq.params}")
            return [], []

    def _get_all_projects(self) -> list[ProjectResult]:
        """Return all projects with their core data, including rooms embedded in units."""
        cypher = """
        MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
        OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
        OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
        WITH p, n, c, dev,
             collect(DISTINCT CASE WHEN u IS NOT NULL THEN u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] } ELSE null END) AS units,
             collect(DISTINCT am.name) AS amenities,
             collect(DISTINCT lm.name) AS landmarks
        RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
               units, amenities, landmarks
        LIMIT $limit
        """
        with self._driver.session() as session:
            records = session.run(cypher, limit=settings.GRAPH_MAX_RESULTS)
            return [self._record_to_result(r) for r in records]

    def _get_filtered_projects(self, intent: QueryIntent) -> list[ProjectResult]:
        """Build a filtered Cypher query from the intent."""
        where_clauses = []
        params: dict = {}

        # Location filtering
        if intent.neighbourhood:
            if isinstance(intent.neighbourhood, list):
                # Multiple neighbourhoods — also search p.address for sub-localities
                conds = []
                for i, nbh in enumerate(intent.neighbourhood):
                    key = f"nbh_{i}"
                    # Normalize hyphen variants: 'New Nikol-Naroda Road' ↔ 'New Nikol - Naroda Road'
                    # REPLACE(n.name, ' - ', '-') normalizes spaces-around-dash both ways
                    conds.append(
                        f"(toLower(n.name) CONTAINS toLower(${key}) "
                        f"OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower(${key}) "
                        f"OR toLower(p.address) CONTAINS toLower(${key}) "
                        f"OR EXISTS {{ MATCH (n)-[:ALIAS_OF*1..2]->(canonical) WHERE toLower(canonical.name) CONTAINS toLower(${key}) }})"
                    )
                    params[key] = nbh
                where_clauses.append(f"({' OR '.join(conds)})")
            else:
                where_clauses.append(
                    "(toLower(n.name) CONTAINS toLower($neighbourhood) "
                    "OR REPLACE(toLower(n.name), ' - ', '-') CONTAINS toLower($neighbourhood) "
                    "OR toLower(p.address) CONTAINS toLower($neighbourhood) "
                    "OR EXISTS { MATCH (n)-[:ALIAS_OF*1..2]->(canonical) WHERE toLower(canonical.name) CONTAINS toLower($neighbourhood) })"
                )
                params["neighbourhood"] = intent.neighbourhood

        if intent.city and not intent.neighbourhood:
            if isinstance(intent.city, list):
                # We need case-insensitive exact match against a list of cities
                # e.g., toLower(c.name) IN [toLower($c1), toLower($c2)]
                city_conds = []
                for i, cty in enumerate(intent.city):
                    key = f"city_{i}"
                    city_conds.append(f"toLower(c.name) = toLower(${key})")
                    params[key] = cty
                where_clauses.append(f"({' OR '.join(city_conds)})")
            else:
                where_clauses.append("toLower(c.name) = toLower($city)")
                params["city"] = intent.city

        if intent.zone:
            if isinstance(intent.zone, list):
                conds = []
                for i, z in enumerate(intent.zone):
                    key = f"zone_{i}"
                    conds.append(f"EXISTS {{ MATCH (n)-[:PART_OF]->(z:Zone) WHERE toLower(z.name) CONTAINS toLower(${key}) }}")
                    params[key] = z
                where_clauses.append(f"({' OR '.join(conds)})")
            else:
                where_clauses.append(
                    "EXISTS { MATCH (n)-[:PART_OF]->(z:Zone) WHERE toLower(z.name) CONTAINS toLower($zone) }"
                )
                params["zone"] = intent.zone

        # BHK filtering
        if intent.bhk is not None:
            if isinstance(intent.bhk, list):
                where_clauses.append(
                    "ANY(u IN units WHERE u.bhk IN $bhk)"
                )
                params["bhk"] = intent.bhk
            else:
                where_clauses.append(
                    "ANY(u IN units WHERE u.bhk = $bhk)"
                )
                params["bhk"] = intent.bhk

        # Property type filtering — exclude completely wrong types
        # e.g. user asks for PENTHOUSE → don't return VILLAs or TEEMENTs
        if intent.property_type:
            if isinstance(intent.property_type, list):
                pt_list = [pt.upper() for pt in intent.property_type]
                where_clauses.append(
                    "ANY(u IN units WHERE toLower(u.property_type) IN [x IN $property_types | toLower(x)])"
                )
                params["property_types"] = pt_list
            else:
                where_clauses.append(
                    "ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))"
                )
                params["property_type"] = intent.property_type.upper()

        # Amenity filtering — check HAS_AMENITY nodes AND project-level boolean flags
        if intent.amenities:
            for i, am in enumerate(intent.amenities):
                key = f"amenity_{i}"
                # Try to find a matching project boolean flag for well-known amenities
                flag_col = _amenity_flag(am)
                if flag_col:
                    where_clauses.append(
                        f"(ANY(am IN amenities WHERE toLower(am) CONTAINS toLower(${key})) "
                        f"OR {flag_col} = 1)"
                    )
                else:
                    where_clauses.append(
                        f"ANY(am IN amenities WHERE toLower(am) CONTAINS toLower(${key}))"
                    )
                params[key] = am

        # Has balcony
        if intent.has_balcony:
            where_clauses.append(
                "ANY(u IN units WHERE u.balcony_sqft IS NOT NULL AND u.balcony_sqft > 0)"
            )

        # Has parking
        if intent.has_parking:
            where_clauses.append("p.has_parking = 1")

        # Specific project names
        if intent.project_names:
            proj_conds = []
            for i, p_name in enumerate(intent.project_names):
                key = f"p_name_{i}"
                proj_conds.append(f"toLower(p.project_name) CONTAINS toLower(${key})")
                params[key] = p_name
            where_clauses.append(f"({' OR '.join(proj_conds)})")

        # Developer
        if intent.developer:
            if isinstance(intent.developer, list):
                dev_conds = []
                for i, dev in enumerate(intent.developer):
                    key = f"dev_{i}"
                    dev_conds.append(f"toLower(dev.name) CONTAINS toLower(${key})")
                    params[key] = dev
                where_clauses.append(f"({' OR '.join(dev_conds)})")
            else:
                where_clauses.append("toLower(dev.name) CONTAINS toLower($developer)")
                params["developer"] = intent.developer

        # ── Area range — qualifier-aware (carpet / super_builtup / both) + around ±15% ──
        _AREA_TOLERANCE = 0.15
        area_qualifier = getattr(intent, "area_qualifier", None)   # "carpet" | "super_builtup" | None
        around_area    = getattr(intent, "around_area", False)      # True → ±15% BETWEEN window

        if intent.min_sqft is not None or intent.max_sqft is not None:

            # Determine lo/hi bounds
            if around_area and intent.min_sqft is not None and intent.max_sqft is None:
                # "around N sqft" — build symmetric ±15% window from min_sqft as the target
                target = float(intent.min_sqft)
                lo: Optional[float] = round(target * (1 - _AREA_TOLERANCE), 2)
                hi: Optional[float] = round(target * (1 + _AREA_TOLERANCE), 2)
            else:
                lo = float(intent.min_sqft) if intent.min_sqft is not None else None
                hi = float(intent.max_sqft) if intent.max_sqft is not None else None

            if lo is not None:
                params["area_lo"] = lo
            if hi is not None:
                params["area_hi"] = hi

            def _field_cond(field: str) -> str:
                """Build the IS NOT NULL + range sub-expression for one field."""
                parts = [f"u.{field} IS NOT NULL"]
                if lo is not None:
                    parts.append(f"toFloat(u.{field}) >= $area_lo")
                if hi is not None:
                    parts.append(f"toFloat(u.{field}) <= $area_hi")
                return " AND ".join(parts)

            carpet_cond      = _field_cond("carpet_sqft")
            super_builtup_cond = _field_cond("super_builtup_sqft")

            if area_qualifier == "carpet":
                where_clauses.append(f"ANY(u IN units WHERE {carpet_cond})")
            elif area_qualifier == "super_builtup":
                where_clauses.append(f"ANY(u IN units WHERE {super_builtup_cond})")
            else:
                # No qualifier — match if EITHER area field satisfies the range
                where_clauses.append(
                    f"ANY(u IN units WHERE ({carpet_cond}) OR ({super_builtup_cond}))"
                )

        # Units per floor filtering
        if intent.min_units_per_floor is not None:
            where_clauses.append(
                "ANY(f IN floor_layouts WHERE f.total_units_on_floor >= $min_units_per_floor)"
            )
            params["min_units_per_floor"] = intent.min_units_per_floor

        if intent.max_units_per_floor is not None:
            where_clauses.append(
                "ANY(f IN floor_layouts WHERE f.total_units_on_floor <= $max_units_per_floor)"
            )
            params["max_units_per_floor"] = intent.max_units_per_floor

        # Landmark type filtering
        landmark_match = ""
        if intent.landmark_types:
            landmark_conditions = " OR ".join(
                [f"lm.landmark_type = '{lt}'" for lt in intent.landmark_types]
            )
            landmark_match = f"AND ({landmark_conditions})"

        # Specific landmark filtering — also search p.address
        specific_lm_clause = ""
        if intent.specific_landmarks:
            for i, lm in enumerate(intent.specific_landmarks):
                key = f"slm_{i}"
                params[key] = lm
                specific_lm_clause += f" OR toLower(lm.name) CONTAINS toLower(${key}) OR toLower(p.address) CONTAINS toLower(${key})"

        cypher = f"""
        MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
        OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout)
        OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
        OPTIONAL MATCH (p)-[:NEAR]->(lm:Landmark)
            {f"WHERE lm.landmark_type IN {json.dumps(intent.landmark_types)}" if intent.landmark_types else ""}
        WITH
            p, n, c,
            dev,
            collect(DISTINCT properties(u)) AS units,
            collect(DISTINCT properties(f)) AS floor_layouts,
            collect(DISTINCT am.name) AS amenities,
            collect(DISTINCT lm.name) AS landmarks
        {"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
        RETURN p, n.name AS neighbourhood, c.name AS city, dev.name AS developer,
               units, floor_layouts, amenities, landmarks
        LIMIT $limit
        """
        params["limit"] = settings.GRAPH_MAX_RESULTS

        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                return [self._record_to_result(r) for r in records]
        except Exception as e:
            logger.warning(f"Filtered Cypher failed ({e}).")
            if where_clauses:
                logger.warning("Strict filters were applied. Returning empty results instead of ignoring filters.")
                return []
            return self._get_all_projects()

    @staticmethod
    def _record_to_result(record) -> ProjectResult:
        p = dict(record["p"])
        units_raw = record.get("units") or []
        # Units may contain embedded rooms (from list comprehension in Cypher)
        units = []
        for u in units_raw:
            if u is None:
                continue
            u_dict = dict(u)
            # Ensure rooms list is a plain list of dicts
            if "rooms" in u_dict and u_dict["rooms"] is not None:
                u_dict["rooms"] = [dict(r) for r in u_dict["rooms"] if r]
            units.append(u_dict)

        floor_layouts_raw = record.get("floor_layouts") or []
        floor_layouts = [dict(f) for f in floor_layouts_raw if f]
        amenities = [a for a in (record.get("amenities") or []) if a]
        landmarks = [lm for lm in (record.get("landmarks") or []) if lm]
        developer = record.get("developer") or p.get("developer_name", "Unknown")
        city = record.get("city") or ""
        neighbourhood = record.get("neighbourhood") or ""

        return ProjectResult(
            project_id=p.get("project_id", ""),
            project_name=p.get("project_name", "Unknown Project"),
            city=city,
            neighbourhood=neighbourhood,
            developer=developer,
            units=units,
            floor_layouts=floor_layouts,
            amenities=amenities,
            landmarks=landmarks,
            extra_props={k: v for k, v in p.items()
                         if k not in ("project_id", "project_name", "developer_name")},
            source="graph",
        )


# ── Vector / ChromaDB Retriever ────────────────────────────────────────────────

class VectorRetriever:
    """Retrieves projects using ChromaDB semantic similarity search + cross-encoder re-ranking."""

    def __init__(self):
        client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.CHROMA_EMBED_MODEL
        )
        self._col = client.get_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        logger.debug(f"VectorRetriever: ChromaDB collection has {self._col.count()} docs.")

        # Load cross-encoder re-ranker at startup only when ENABLE_RERANKER=true.
        # Set ENABLE_RERANKER=false in .env to skip model loading entirely (saves
        # memory + startup time). Re-enable any time without code changes.
        self._reranker = None
        if settings.ENABLE_RERANKER:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(settings.RERANK_MODEL)
                logger.info(f"Cross-encoder re-ranker loaded at startup: {settings.RERANK_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder ({e}). Re-ranking disabled.")
                self._reranker = False  # sentinel => skip re-ranking
        else:
            self._reranker = False  # deliberately disabled via config
            logger.info(
                "Cross-encoder re-ranker DISABLED (ENABLE_RERANKER=false). "
                "Using distance-sorted ChromaDB ranking + RERANK_TOP_K cap instead."
            )

    def _build_query_text(self, intent: QueryIntent, query_override: Optional[str] = None) -> str:
        """Build a natural-language query sentence from intent for semantic search."""
        if query_override:
            return query_override

        parts: list[str] = []

        # Lead with the unit type
        if intent.bhk is not None:
            if isinstance(intent.bhk, list):
                parts.append(" or ".join(f"{b} BHK" for b in intent.bhk))
            else:
                parts.append(f"{intent.bhk} BHK")

        if intent.property_type:
            if isinstance(intent.property_type, list):
                parts.append(" or ".join(pt.lower() for pt in intent.property_type))
            else:
                parts.append(intent.property_type.lower())
        elif parts:
            parts.append("residential property")

        # Location
        loc_parts: list[str] = []
        if intent.neighbourhood:
            if isinstance(intent.neighbourhood, list):
                loc_parts.extend(intent.neighbourhood)
            else:
                loc_parts.append(intent.neighbourhood)
        if intent.city:
            if isinstance(intent.city, list):
                loc_parts.extend(intent.city)
            else:
                loc_parts.append(intent.city)
        if loc_parts:
            parts.append(f"in {', '.join(loc_parts)}")

        # Amenities
        if intent.amenities:
            parts.append(f"with {', '.join(intent.amenities)}")

        # Landmarks
        if intent.specific_landmarks:
            parts.append(f"near {', '.join(intent.specific_landmarks)}")

        # Project names
        if intent.project_names:
            parts.append(f"project {', '.join(intent.project_names)}")

        # Semantic/fuzzy keywords
        if intent.semantic_keywords:
            parts.append(", ".join(intent.semantic_keywords))

        # Facing
        if intent.entrance_facing:
            parts.append(f"{intent.entrance_facing} facing")

        return " ".join(parts) if parts else "residential apartment project"

    def _build_where_filter(self, intent: QueryIntent, doc_type: Optional[str] = None) -> Optional[dict]:
        """Build ChromaDB metadata filter from intent fields."""
        conditions: list[dict] = []

        if doc_type:
            conditions.append({"doc_type": {"$eq": doc_type}})

        if intent.bhk is not None and doc_type != "project":
            if isinstance(intent.bhk, list):
                conditions.append({"bhk": {"$in": intent.bhk}})
            else:
                conditions.append({"bhk": {"$eq": intent.bhk}})

        if intent.city:
            if isinstance(intent.city, list):
                conditions.append({"city": {"$in": intent.city}})
            else:
                conditions.append({"city": {"$eq": intent.city}})

        # Amenity boolean flags — only for well-known amenities
        if intent.amenities:
            for am in intent.amenities:
                am_lower = am.lower().strip()
                if any(kw in am_lower for kw in ("pool", "swimming")):
                    conditions.append({"has_pool": {"$eq": 1}})
                elif any(kw in am_lower for kw in ("clubhouse", "club house", "club")):
                    conditions.append({"has_clubhouse": {"$eq": 1}})
                elif any(kw in am_lower for kw in ("park", "garden")):
                    conditions.append({"has_park": {"$eq": 1}})
                elif any(kw in am_lower for kw in ("parking", "car park")):
                    conditions.append({"has_parking": {"$eq": 1}})

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def retrieve(self, intent: QueryIntent, query_override: Optional[str] = None) -> list[ProjectResult]:
        """
        Two-phase vector retrieval:
          Phase 1: ChromaDB semantic search (project + unit docs, metadata filtered)
          Phase 2: Cross-encoder re-ranking for precision

        Strategy:
          - For queries with BHK/area/room specifics → search unit docs primarily
          - For broad/amenity/fuzzy queries → search project docs primarily
          - Always search both for comprehensive coverage
        """
        query_text = self._build_query_text(intent, query_override)
        logger.info(f"Vector query text: {query_text!r}")

        # Determine if this is a unit-specific or broad query
        is_unit_specific = (intent.bhk is not None or intent.min_sqft or intent.max_sqft
                           or intent.entrance_facing)

        # ── Phase 1: Retrieve candidates from both doc types ──
        all_hits: list[tuple[dict, float, str]] = []  # (metadata, distance, document)

        # Search unit docs (always, but prioritize for unit-specific queries)
        unit_k = settings.VECTOR_TOP_K if is_unit_specific else max(10, settings.VECTOR_TOP_K // 2)
        unit_filter = self._build_where_filter(intent, doc_type="unit")
        unit_hits = self._query_chroma(query_text, unit_k, unit_filter)
        if not unit_hits and unit_filter:
            # Fallback: try with just doc_type filter
            unit_hits = self._query_chroma(query_text, unit_k, {"doc_type": {"$eq": "unit"}})
        all_hits.extend(unit_hits)

        # Search project docs (always, but prioritize for broad queries)
        proj_k = max(10, settings.VECTOR_TOP_K // 2) if is_unit_specific else settings.VECTOR_TOP_K
        proj_filter = self._build_where_filter(intent, doc_type="project")
        proj_hits = self._query_chroma(query_text, proj_k, proj_filter)
        if not proj_hits and proj_filter:
            proj_hits = self._query_chroma(query_text, proj_k, {"doc_type": {"$eq": "project"}})
        all_hits.extend(proj_hits)

        if not all_hits:
            logger.info("Vector retrieval → 0 results")
            return []

        # ── Phase 1.5: Distance threshold filtering ──
        pre_threshold_count = len(all_hits)
        all_hits = [
            (meta, dist, doc) for meta, dist, doc in all_hits
            if dist <= settings.VECTOR_DISTANCE_THRESHOLD
        ]
        if len(all_hits) < pre_threshold_count:
            logger.info(
                f"Distance threshold ({settings.VECTOR_DISTANCE_THRESHOLD}): "
                f"filtered {pre_threshold_count} → {len(all_hits)} hits"
            )
        if not all_hits:
            logger.info("Vector retrieval → 0 results after distance threshold")
            return []

        # ── Phase 2: Re-ranking / candidate capping ────────────────────────────
        # RERANK_TOP_K always caps how many unique projects reach the LLM judge.
        # This is the primary control for judge token budget — whether or not the
        # cross-encoder is enabled.
        reranker = self._reranker if self._reranker is not False else None

        if reranker and len(all_hits) > 3:
            # ── Cross-encoder path (ENABLE_RERANKER=true) ──────────────────────
            # Sort by distance first, cap candidates to avoid excessive CPU time.
            MAX_RERANK_CANDIDATES = 20
            if len(all_hits) > MAX_RERANK_CANDIDATES:
                all_hits.sort(key=lambda x: x[1])  # sort by distance (lower=better)
                all_hits = all_hits[:MAX_RERANK_CANDIDATES]
                logger.info(f"Capped re-rank candidates to {MAX_RERANK_CANDIDATES} (from {pre_threshold_count})")
            try:
                pairs = [(query_text, doc) for _, _, doc in all_hits]
                scores = reranker.predict(pairs, batch_size=32)
                # Combine: sort by cross-encoder score (higher = more relevant)
                ranked = sorted(
                    zip(scores, all_hits),
                    key=lambda x: x[0],
                    reverse=True,
                )
                # Apply re-rank score threshold
                ranked = [
                    (score, hit) for score, hit in ranked
                    if score >= settings.RERANK_SCORE_THRESHOLD
                ]
                if not ranked:
                    logger.info("All results filtered by re-rank score threshold")
                    return []
                all_hits = [hit for _, hit in ranked[:settings.RERANK_TOP_K]]
                logger.info(
                    f"Cross-encoder: re-ranked {len(pairs)} → kept top {len(all_hits)} "
                    f"(score threshold: {settings.RERANK_SCORE_THRESHOLD})"
                )
            except Exception as e:
                logger.warning(f"Re-ranking failed ({e}), using raw distances")
                all_hits.sort(key=lambda x: x[1])
                all_hits = all_hits[:settings.RERANK_TOP_K]
        else:
            # ── Distance-sorted path (ENABLE_RERANKER=false or too few hits) ────
            # ChromaDB L2 distance is already a strong relevance signal:
            # lower distance = more semantically similar to the query.
            # Sort ascending (most relevant first) and cap at RERANK_TOP_K.
            all_hits.sort(key=lambda x: x[1])   # ascending: closest = most relevant
            all_hits = all_hits[:settings.RERANK_TOP_K]
            logger.info(
                f"Distance-ranked: kept top {len(all_hits)} candidates "
                f"(RERANK_TOP_K={settings.RERANK_TOP_K}, threshold={settings.VECTOR_DISTANCE_THRESHOLD})"
            )

        # ── Assemble ProjectResults, grouped by project_id ──
        project_map: dict[str, ProjectResult] = {}
        for meta, dist, doc in all_hits:
            pid = meta.get("project_id", "")
            if pid not in project_map:
                amenities_str = meta.get("amenities", "")
                amenities_list = [a.strip() for a in amenities_str.split(",") if a.strip()] if amenities_str else []
                project_map[pid] = ProjectResult(
                    project_id=pid,
                    project_name=meta.get("project_name", "Unknown"),
                    city=meta.get("city", ""),
                    neighbourhood=meta.get("neighbourhood", ""),
                    developer=meta.get("developer", ""),
                    units=[],
                    score=dist,
                    source="vector",
                    amenities=amenities_list,
                    extra_props={
                        "project_status": meta.get("project_status", ""),
                        "has_clubhouse": meta.get("has_clubhouse", 0),
                        "has_pool": meta.get("has_pool", 0),
                        "has_park": meta.get("has_park", 0),
                        "has_parking": meta.get("has_parking", 0),
                    },
                )

            # Add unit info from unit-type docs
            if meta.get("doc_type") == "unit":
                bhk = meta.get("bhk", 0)
                ptype = meta.get("property_type", "")
                area = meta.get("area_sqft", 0.0)
                ut = meta.get("unit_type", "")
                if bhk > 0 or ptype:
                    utyp = ut if ut else (f"{bhk} BHK" if bhk > 0 else "Unit")
                    udict: dict = {"unit_type": utyp, "property_type": ptype}
                    if area > 0.0:
                        udict["super_builtup_sqft"] = area
                    facing = meta.get("entrance_facing", "")
                    if facing:
                        udict["entrance_facing"] = facing
                    # Deduplicate
                    if udict not in project_map[pid].units:
                        project_map[pid].units.append(udict)

            # Track best score
            if dist < project_map[pid].score:
                project_map[pid].score = dist

        sorted_results = sorted(project_map.values(), key=lambda r: r.score)
        logger.info(f"Vector retrieval → {len(sorted_results)} project(s)")
        return sorted_results

    def _query_chroma(
        self, query_text: str, n_results: int, where_filter: Optional[dict]
    ) -> list[tuple[dict, float, str]]:
        """Execute a ChromaDB query and return (metadata, distance, document) tuples."""
        try:
            actual_n = min(n_results, self._col.count())
            if actual_n <= 0:
                return []
            kwargs = {
                "query_texts": [query_text],
                "n_results": actual_n,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = self._col.query(**kwargs)

            if not results["ids"] or len(results["ids"][0]) == 0:
                return []

            hits: list[tuple[dict, float, str]] = []
            for meta, dist, doc in zip(
                results["metadatas"][0], results["distances"][0], results["documents"][0]
            ):
                hits.append((meta, dist, doc))
            return hits

        except Exception as e:
            logger.warning(f"ChromaDB query failed ({e})")
            return []


# ── Answer data formatter ──────────────────────────────────────────────────────

def _format_answer_data(
    answer_data: list[dict],
    query_type: str,
    raw_query: str,
    params: dict,
) -> str:
    """
    Convert raw answer_data from LOOKUP / AGGREGATE Cypher into a
    human-readable direct answer string.

    Handles multiple answer formats:
    - Room-specific LOOKUP (rooms per unit variant)
    - Full project detail LOOKUP (all units + all rooms)
    - Project info LOOKUP (address, RERA, status, etc.)
    - AGGREGATE with size filter or existence filter
    - Count/statistics AGGREGATE
    """
    if not answer_data:
        return ""

    lines: list[str] = []
    room_name_param = params.get("room_name", "")

    # ── Detect answer format ──
    first = answer_data[0] if answer_data else {}

    # Case: Project info dict (has 'project_name' + 'address' or similar top-level keys)
    if isinstance(first, dict) and "project_name" in first and ("address" in first or "rera_number" in first or "project_status" in first):
        lines.append("📋 **Project Information:**")
        for entry in answer_data:
            if not isinstance(entry, dict):
                continue
            for key, val in entry.items():
                if val is None or val == "" or val == []:
                    continue
                readable_key = key.replace("_", " ").title()
                if isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val)
                else:
                    val_str = str(val)
                lines.append(f"  - **{readable_key}**: {val_str}")
        return "\n".join(lines) if lines else ""

    # Case: Count/statistics dict (has 'count' key)
    if isinstance(first, dict) and "count" in first:
        count = first.get("count", 0)
        projects = first.get("projects", [])
        lines.append(f"📊 **Count: {count}**")
        if projects and len(projects) <= 20:
            lines.append("Projects: " + ", ".join(str(p) for p in projects))
        return "\n".join(lines)

    # Case: Room-level results (LOOKUP or AGGREGATE)
    if query_type == "LOOKUP":
        lines.append("📐 **Direct Answer:**")
        for entry in answer_data:
            if not isinstance(entry, dict):
                continue
            unit_type = entry.get("unit_type") or entry.get("bhk", "")
            if unit_type:
                lines.append(f"\n**{unit_type}:**")

            # Show unit-level details if present
            carpet = entry.get("carpet_sqft")
            sba = entry.get("super_builtup_sqft")
            facing = entry.get("entrance_facing")
            desc = entry.get("description")
            prop_type = entry.get("property_type")
            
            detail_parts = []
            if prop_type:
                detail_parts.append(f"Type: {prop_type}")
            if carpet:
                try:
                    detail_parts.append(f"Carpet: {float(carpet):.1f} sqft")
                except (ValueError, TypeError):
                    detail_parts.append(f"Carpet: {carpet} sqft")
            if sba:
                try:
                    detail_parts.append(f"Super Built-up: {float(sba):.1f} sqft")
                except (ValueError, TypeError):
                    detail_parts.append(f"Super Built-up: {sba} sqft")
            if facing:
                detail_parts.append(f"Facing: {facing}")
            if detail_parts:
                lines.append(f"  {' | '.join(detail_parts)}")
            if desc:
                truncated = desc[:150] + ("…" if len(desc) > 150 else "")
                lines.append(f"  _{truncated}_")

            rooms = entry.get("rooms") or []
            if not rooms:
                rooms = entry.get("matching_rooms") or []

            if not rooms:
                if room_name_param:
                    lines.append(f"  - {room_name_param}: dimensions not available for this unit")
                continue

            for r in rooms:
                if not r:
                    continue
                rname = r.get("room_name") or r.get("name") or room_name_param or "Room"
                length = r.get("length")
                width  = r.get("width")
                area   = r.get("area_sqft")
                parts  = []
                if length and width:
                    parts.append(f"{length} x {width}")
                if area:
                    try:
                        parts.append(f"{float(area):.1f} sqft")
                    except (ValueError, TypeError):
                        parts.append(f"{area} sqft")
                dim_str = " = " + ", ".join(parts) if parts else " (dimensions not provided)"
                lines.append(f"  - {rname}{dim_str}")

    elif query_type == "AGGREGATE":
        has_results = any(
            (entry.get("matching_rooms") or []) for entry in answer_data
            if isinstance(entry, dict)
        )
        if not has_results:
            return ""

        min_area = params.get("min_area")
        max_area = params.get("max_area")
        area_min = params.get("area_min")  # LLM "around" style
        area_max = params.get("area_max")  # LLM "around" style
        if area_min is not None and area_max is not None:
            filter_desc = f"~{area_min}–{area_max} sqft (±15%)"
        elif min_area is not None and max_area is not None:
            filter_desc = f"~{min_area}–{max_area} sqft (±15%)"
        elif min_area is not None:
            filter_desc = f"> {min_area} sqft"
        elif max_area is not None:
            filter_desc = f"< {max_area} sqft"
        elif area_min is not None:
            filter_desc = f">= {area_min} sqft"
        elif area_max is not None:
            filter_desc = f"<= {area_max} sqft"
        else:
            filter_desc = "available"

        room_display = room_name_param or "Room"
        lines.append(f"📐 **{room_display} sizes ({filter_desc}) — per unit type:**")

        seen: set[str] = set()
        for entry in answer_data:
            if not isinstance(entry, dict):
                continue
            # entry is the outer project dict: {project_name, matching_rooms: [...]}
            matching = entry.get("matching_rooms") or []
            for item in matching:
                if not item:
                    continue
                # Handle both formats LLM may generate:
                # NEW flat:  {unit_type, bhk, room_name, area_sqft, length, width}
                # OLD wrapped: {unit_type, bhk, rooms: [{room_name, area_sqft, ...}]}
                if "rooms" in item and isinstance(item.get("rooms"), list):
                    # Old unit-wrapped format — iterate inner rooms sub-list
                    unit_type = str(item.get("unit_type") or item.get("bhk") or "")
                    for r in item["rooms"]:
                        if not r:
                            continue
                        rname  = r.get("room_name") or r.get("name") or room_display
                        length = r.get("length")
                        width  = r.get("width")
                        area   = r.get("area_sqft")
                        parts: list[str] = []
                        if length and width:
                            parts.append(f"{length} x {width}")
                        if area:
                            try:
                                parts.append(f"{float(area):.1f} sqft")
                            except (ValueError, TypeError):
                                parts.append(f"{area} sqft")
                        key = f"{unit_type}|{rname}|{'|'.join(parts)}"
                        if key in seen:
                            continue
                        seen.add(key)
                        dim_str = ": " + ", ".join(parts) if parts else ""
                        unit_prefix = f"[{unit_type}] " if unit_type else ""
                        # lines.append(f"  - {unit_prefix}{rname}{dim_str}")
                else:
                    # New flat format: room_name + unit_type are direct fields
                    unit_type = str(item.get("unit_type") or item.get("bhk") or "")
                    rname  = item.get("room_name") or item.get("name") or room_display
                    length = item.get("length")
                    width  = item.get("width")
                    area   = item.get("area_sqft")
                    parts = []
                    if length and width:
                        parts.append(f"{length} x {width}")
                    if area:
                        try:
                            parts.append(f"{float(area):.1f} sqft")
                        except (ValueError, TypeError):
                            parts.append(f"{area} sqft")
                    key = f"{unit_type}|{rname}|{'|'.join(parts)}"
                    if key in seen:
                        continue
                    seen.add(key)
                    dim_str = ": " + ", ".join(parts) if parts else ""
                    unit_prefix = f"[{unit_type}] " if unit_type else ""
                    # lines.append(f"  - {unit_prefix}{rname}{dim_str}")

    return "\n".join(lines) if lines else ""


# ── Dual Retriever (main entry point) ─────────────────────────────────────────

class DualRetriever:
    """
    Orchestrates both graph and vector retrieval, merges results,
    and assembles the final context string.
    """

    def __init__(self):
        self._graph = GraphRetriever()
        self._vector = VectorRetriever()

    def close(self):
        self._graph.close()

    # ── Neo4j enrichment for vector-only results ───────────────────────────
    def _enrich_from_graph(self, vector_only: list[ProjectResult]) -> None:
        """Batch-fetch full project data from Neo4j for vector-only results.

        Runs a single Cypher query for ALL vector-only project IDs,
        then fills in the missing fields (address, rooms, amenities,
        landmarks, society_description, etc.) in-place.
        """
        if not vector_only:
            return

        pids = [r.project_id for r in vector_only]
        cypher = """
        MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
        WHERE p.project_id IN $pids
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
        WITH p, n, c, dev, units, amenities, landmarks
        RETURN p, n.name AS neighbourhood, c.name AS city,
               dev.name AS developer, units, amenities, landmarks
        """
        try:
            with self._graph._driver.session() as session:
                records = list(session.run(cypher, pids=pids))
        except Exception as e:
            logger.warning(f"Vector enrichment query failed ({e}). Cards may be sparse.")
            return

        # Build a lookup by project_id
        enriched: dict[str, dict] = {}
        for rec in records:
            p = dict(rec["p"])
            pid = p.get("project_id", "")
            units_raw = rec.get("units") or []
            units_clean = []
            for u in units_raw:
                if u is None:
                    continue
                u_dict = dict(u)
                if "rooms" in u_dict and u_dict["rooms"] is not None:
                    u_dict["rooms"] = [dict(r) for r in u_dict["rooms"] if r]
                units_clean.append(u_dict)
            enriched[pid] = {
                "neighbourhood": rec.get("neighbourhood") or "",
                "city": rec.get("city") or "",
                "developer": rec.get("developer") or p.get("developer_name", ""),
                "units": units_clean,
                "amenities": [a for a in (rec.get("amenities") or []) if a],
                "landmarks": [lm for lm in (rec.get("landmarks") or []) if lm],
                "extra_props": {k: v for k, v in p.items()
                                if k not in ("project_id", "project_name", "developer_name")},
            }

        # Patch each vector-only result in-place
        for r in vector_only:
            data = enriched.get(r.project_id)
            if not data:
                continue
            r.neighbourhood = data["neighbourhood"] or r.neighbourhood
            r.city = data["city"] or r.city
            r.developer = data["developer"] or r.developer
            if data["units"]:
                r.units = data["units"]
            if data["amenities"]:
                r.amenities = data["amenities"]
            if data["landmarks"]:
                r.landmarks = data["landmarks"]
            # Merge extra_props (Neo4j data wins for any key present)
            r.extra_props = {**r.extra_props, **data["extra_props"]}

        logger.info(f"Enriched {len(enriched)}/{len(vector_only)} vector-only result(s) from Neo4j")

    def retrieve_and_assemble(
        self, intent: QueryIntent, raw_query: str,
        cypher_result: CypherQuery | None = None,
    ) -> tuple[str, list[ProjectResult], str]:
        """
        Main retrieval method.

        Args:
            intent: The structured QueryIntent (from cypher_result.intent or standalone).
            raw_query: The raw user query string.
            cypher_result: Pre-generated CypherQuery (if None, will be generated here).

        Returns:
            (context_text, merged_project_results, direct_answer_text)
            - direct_answer_text is non-empty for LOOKUP / AGGREGATE queries
              and contains a human-readable answer to the user's specific question.
        """
        # ── Cache check: return instantly for repeated identical queries ──────
        cache_key = raw_query.lower().strip()
        now = time.time()
        if cache_key in _QUERY_CACHE:
            cached_ts, cached_result = _QUERY_CACHE[cache_key]
            if now - cached_ts < _CACHE_TTL_SECONDS:
                logger.info(f"[Cache] HIT for query: {raw_query!r} (age: {int(now - cached_ts)}s)")
                return cached_result
            else:
                del _QUERY_CACHE[cache_key]  # expired — remove and re-run

        # 1. Generate precise Cypher + vector query string from the raw user query
        if cypher_result is None:
            cypher_result = generate_cypher(raw_query)

        # 2. Launch Graph + Vector retrieval in PARALLEL
        def _do_graph():
            return self._graph.retrieve(cypher_result)

        def _do_vector():
            try:
                return self._vector.retrieve(
                    intent,
                    query_override=cypher_result.vector_query,
                )
            except Exception as e:
                logger.warning(f"Vector retrieval failed ({e}). Proceeding with graph only.")
                return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            graph_future: Future = executor.submit(_do_graph)
            vector_future: Future = executor.submit(_do_vector)

            graph_results, answer_data = graph_future.result()
            vector_results = vector_future.result()

        # 3. Score-based merge
        #    Graph results are canonical (score = 0.0 baseline)
        #    Vector results carry their semantic similarity score
        #    Projects found in BOTH sources get a boost
        merged: dict[str, ProjectResult] = {}

        for r in graph_results:
            r.score = 0.0  # graph matches are structurally perfect
            merged[r.project_id] = r

        for r in vector_results:
            if r.project_id not in merged:
                # Vector-only result — keep as-is
                merged[r.project_id] = r
            else:
                # Found in BOTH graph and vector — boost!
                merged[r.project_id].source = "both"
                # Keep the graph result's richer data, but note the vector score
                merged[r.project_id].score = -0.5  # boost: "both" scores better than "graph"
                # Merge any extra unit types that vector found but graph didn't
                existing_types = {u.get("unit_type") for u in merged[r.project_id].units}
                for vu in r.units:
                    if vu.get("unit_type") not in existing_types:
                        merged[r.project_id].units.append(vu)

        # Enrich vector-only results with full Neo4j data (single batch query)
        vector_only = [r for r in merged.values() if r.source == "vector"]
        if vector_only:
            self._enrich_from_graph(vector_only)

        # 4. Sort: "both" first (lowest score), then "graph", then "vector"
        def sort_key(r: ProjectResult):
            source_prio = 0 if r.source == "both" else (1 if r.source == "graph" else 2)
            return (source_prio, r.score)

        final_candidates = sorted(merged.values(), key=sort_key)

        # For LOOKUP queries don't apply name filter (Cypher already filtered by project name)
        if intent.project_names and cypher_result.query_type not in ("LOOKUP", "AGGREGATE"):
            filtered_final = []
            for candidate in final_candidates:
                candidate_name = candidate.project_name.lower()
                for requested_name in intent.project_names:
                    if requested_name.lower() in candidate_name or candidate_name in requested_name.lower():
                        filtered_final.append(candidate)
                        break
            final = filtered_final[: settings.FINAL_TOP_N]
            if not final:
                final = final_candidates[: settings.FINAL_TOP_N]
        else:
            final = final_candidates[: settings.FINAL_TOP_N]

        # ── Property-type post-filter ──────────────────────────────────────────
        # If the user explicitly requested a specific property type (e.g. PENTHOUSE,
        # APARTMENT), drop any merged result whose units are ALL of a different type.
        # This prevents villas from appearing in apartment/penthouse searches and
        # vice versa.  A project is kept if it has AT LEAST ONE unit matching the
        # requested type(s), or if its units have no property_type metadata at all
        # (to avoid false negatives from sparse knowledge-graph data).
        if intent.property_type and cypher_result.query_type not in ("LOOKUP", "AGGREGATE"):
            requested_types: list[str]
            if isinstance(intent.property_type, list):
                requested_types = [pt.upper() for pt in intent.property_type]
            else:
                requested_types = [intent.property_type.upper()]

            def _matches_property_type(proj: ProjectResult) -> bool:
                if not proj.units:
                    return True  # no unit metadata — keep to avoid false negatives
                unit_types = [
                    u.get("property_type", "").upper()
                    for u in proj.units
                    if u.get("property_type")
                ]
                if not unit_types:
                    return True  # units exist but no property_type field — keep
                return any(ut in requested_types for ut in unit_types)

            filtered_by_type = [p for p in final if _matches_property_type(p)]
            if filtered_by_type:
                logger.info(
                    f"Property-type filter ({requested_types}): "
                    f"{len(final)} → {len(filtered_by_type)} project(s)"
                )
                final = filtered_by_type
            else:
                # All candidates were filtered out — likely sparse data.
                # Fall back to the original list so we don't return empty.
                logger.info(
                    f"Property-type filter ({requested_types}) would remove all results — "
                    "keeping originals (sparse unit data)."
                )

        # 5. Build direct answer text for LOOKUP / AGGREGATE
        direct_answer_text = ""
        if answer_data and cypher_result.query_type in ("LOOKUP", "AGGREGATE"):
            direct_answer_text = _format_answer_data(
                answer_data,
                query_type=cypher_result.query_type,
                raw_query=raw_query,
                params=cypher_result.params,
            )

        # 6. Assemble context text
        context_blocks = []
        for i, proj in enumerate(final, 1):
            context_blocks.append(f"[{i}] {proj.to_context_text()}")

        context_text = (
            f"User query: {raw_query}\n\n"
            f"Retrieved {len(final)} relevant residential project(s):\n\n"
            + "\n\n".join(context_blocks)
        )

        logger.success(
            f"Context assembled: {len(final)} projects, {len(context_text)} chars, "
            f"direct_answer={'yes' if direct_answer_text else 'no'}"
        )

        # ── Cache store ───────────────────────────────────────────────────────
        result_tuple = (context_text, final, direct_answer_text)
        if len(_QUERY_CACHE) >= _CACHE_MAX_SIZE:
            # Evict the oldest entry
            oldest_key = min(_QUERY_CACHE, key=lambda k: _QUERY_CACHE[k][0])
            del _QUERY_CACHE[oldest_key]
        _QUERY_CACHE[cache_key] = (time.time(), result_tuple)
        logger.info(f"[Cache] STORED query: {raw_query!r} (cache size: {len(_QUERY_CACHE)})")

        return context_text, final, direct_answer_text
