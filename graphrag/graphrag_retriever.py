"""
graphrag_retriever.py — Dual retrieval from Neo4j (graph) + ChromaDB (vector).

Given a QueryIntent, this module:
  1. Runs dynamic Cypher queries against Neo4j to find structurally matching projects
  2. Runs semantic similarity search in ChromaDB for fuzzy/subjective queries
  3. Merges and deduplicates results into a unified context string for the LLM

The merged context is the "retrieved knowledge" passed to graphrag_answer.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from neo4j import GraphDatabase

from graphrag_config import settings
from graphrag_intent import QueryIntent
from graphrag_cypher import generate_cypher, CypherQuery



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
        Falls back to _get_all_projects() for GLOBAL queries or on error.

        Returns:
            (project_results, answer_data)
            - answer_data is non-empty only for LOOKUP / AGGREGATE queries
        """
        if cypher_result.query_type == "GLOBAL":
            results = self._get_all_projects()
            answer_data: list[dict] = []
        else:
            results, answer_data = self._execute_cypher(cypher_result)

        logger.info(f"Graph retrieval → {len(results)} project(s)")
        return results, answer_data

    def _execute_cypher(self, cq: CypherQuery) -> tuple[list[ProjectResult], list[dict]]:
        """Execute the LLM-generated Cypher and return (project_results, answer_data)."""
        try:
            with self._driver.session() as session:
                records = list(session.run(cq.cypher, **cq.params))
                results = [self._record_to_result(r) for r in records]

                # Extract answer_data from LOOKUP / AGGREGATE queries
                answer_data: list[dict] = []
                if cq.answer_columns:
                    for record in records:
                        for col in cq.answer_columns:
                            val = record.get(col)
                            if val:
                                # val is a list of dicts (one per unit)
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
                # Multiple neighbourhoods
                conds = []
                for i, nbh in enumerate(intent.neighbourhood):
                    key = f"nbh_{i}"
                    conds.append(f"(toLower(n.name) CONTAINS toLower(${key}) OR EXISTS {{ MATCH (n)-[:ALIAS_OF*1..2]->(canonical) WHERE toLower(canonical.name) CONTAINS toLower(${key}) }})")
                    params[key] = nbh
                where_clauses.append(f"({' OR '.join(conds)})")
            else:
                where_clauses.append(
                    "(toLower(n.name) CONTAINS toLower($neighbourhood) "
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

        # Area range
        if intent.min_sqft is not None:
            where_clauses.append(
                "ANY(u IN units WHERE u.super_builtup_sqft >= $min_sqft OR u.carpet_sqft >= $min_sqft)"
            )
            params["min_sqft"] = intent.min_sqft

        if intent.max_sqft is not None:
            where_clauses.append(
                "ANY(u IN units WHERE (u.super_builtup_sqft IS NULL OR u.super_builtup_sqft <= $max_sqft) "
                "AND (u.carpet_sqft IS NULL OR u.carpet_sqft <= $max_sqft))"
            )
            params["max_sqft"] = intent.max_sqft

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

        # Specific landmark filtering
        specific_lm_clause = ""
        if intent.specific_landmarks:
            for i, lm in enumerate(intent.specific_landmarks):
                key = f"slm_{i}"
                params[key] = lm
                specific_lm_clause += f" OR toLower(lm.name) CONTAINS toLower(${key})"

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
    """Retrieves units using ChromaDB semantic similarity search."""

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

    def retrieve(self, intent: QueryIntent, query_override: Optional[str] = None) -> list[ProjectResult]:
        """
        Build a natural language query string from the intent (or use query_override)
        and run vector search. Also applies metadata filters where possible (bhk, city).
        """
        # Build a rich query text from intent components
        query_parts = []
        if intent.bhk is not None:
            if isinstance(intent.bhk, list):
                query_parts.extend([f"{b} BHK" for b in intent.bhk])
            else:
                query_parts.append(f"{intent.bhk} BHK")
        if intent.property_type:
            if isinstance(intent.property_type, list):
                query_parts.extend([pt.lower() for pt in intent.property_type])
            else:
                query_parts.append(intent.property_type.lower())
        if intent.neighbourhood:
            if isinstance(intent.neighbourhood, list):
                query_parts.extend(intent.neighbourhood)
            else:
                query_parts.append(intent.neighbourhood)
        if intent.city:
            if isinstance(intent.city, list):
                query_parts.extend(intent.city)
            else:
                query_parts.append(intent.city)
        if intent.amenities:
            query_parts.extend(intent.amenities)
        if intent.semantic_keywords:
            query_parts.extend(intent.semantic_keywords)
        if intent.specific_landmarks:
            query_parts.extend(intent.specific_landmarks)
        if intent.project_names:
            query_parts.extend(intent.project_names)

        # Use query_override from Text-to-Cypher if provided, otherwise build from intent
        if query_override:
            query_text = query_override
        else:
            query_text = " ".join(query_parts) if query_parts else "residential apartment"

        # Metadata filters for ChromaDB (exact match only)
        where_filter = None
        conditions = []
        if intent.bhk is not None:
            if isinstance(intent.bhk, list):
                conditions.append({"bhk": {"$in": intent.bhk}})
            else:
                conditions.append({"bhk": {"$eq": intent.bhk}})
        if intent.city:
            if isinstance(intent.city, list):
                conditions.append({"city": {"$in": intent.city}})
            else:
                conditions.append({"city": {"$eq": intent.city}})
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            kwargs = {
                "query_texts": [query_text],
                "n_results": min(settings.VECTOR_TOP_K, self._col.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = self._col.query(**kwargs)
            
            # If a strict metadata filter returned 0 results (e.g. city="Surat"),
            # do NOT fall back to unfiltered search. Return empty straight away.
            if where_filter and (not results["ids"] or len(results["ids"][0]) == 0):
                logger.info("Vector query with strict filter returned 0 results. Respecting filter.")
                return []
                
        except Exception as e:
            logger.warning(f"Vector query with filter failed ({e}). Returning empty results instead of ignoring filter.")
            return []

        # Group hits by project_id (multiple units per project)
        project_map: dict[str, ProjectResult] = {}
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        for meta, dist, doc in zip(metadatas, distances, documents):
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
                    amenities=amenities_list
                )
            
            # Reconstruct dummy unit layout from vector metadata for UI presentation
            bhk = meta.get("bhk", 0)
            ptype = meta.get("property_type", "")
            area = meta.get("area_sqft", 0.0)
            if bhk > 0 or ptype:
                utyp = f"{bhk} BHK" if bhk > 0 else "Unit"
                udict = {"unit_type": utyp, "property_type": ptype}
                if area > 0.0:
                    udict["super_builtup_sqft"] = area
                
                # Deduplicate units for rendering
                if udict not in project_map[pid].units:
                    project_map[pid].units.append(udict)

            # Keep updating with best score
            if dist < project_map[pid].score:
                project_map[pid].score = dist

        sorted_results = sorted(project_map.values(), key=lambda r: r.score)
        logger.info(f"Vector retrieval → {len(sorted_results)} project(s)")
        return sorted_results


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
        if min_area is not None:
            filter_desc = f"> {min_area} sqft"
        elif max_area is not None:
            filter_desc = f"< {max_area} sqft"
        else:
            filter_desc = "available"

        room_display = room_name_param or "Room"
        lines.append(f"📐 **{room_display} sizes ({filter_desc}) — per unit type:**")

        seen: set[str] = set()
        for entry in answer_data:
            if not isinstance(entry, dict):
                continue
            unit_type = str(entry.get("unit_type") or entry.get("bhk") or "")
            matching = entry.get("matching_rooms") or []
            for r in matching:
                if not r:
                    continue
                rname  = r.get("room_name") or r.get("name") or room_display
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
                key = f"{unit_type}|{rname}|{'|'.join(parts)}"
                if key in seen:
                    continue
                seen.add(key)
                dim_str    = ": " + ", ".join(parts) if parts else ""
                unit_prefix = f"[{unit_type}] " if unit_type else ""
                lines.append(f"  - {unit_prefix}{rname}{dim_str}")

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

    def retrieve_and_assemble(
        self, intent: QueryIntent, raw_query: str
    ) -> tuple[str, list[ProjectResult], str]:
        """
        Main retrieval method.

        Returns:
            (context_text, merged_project_results, direct_answer_text)
            - direct_answer_text is non-empty for LOOKUP / AGGREGATE queries
              and contains a human-readable answer to the user's specific question.
        """
        # 1. Generate precise Cypher + vector query string from the raw user query
        cypher_result = generate_cypher(raw_query)

        # 2. Graph retrieval — precise structural filtering via LLM-generated Cypher
        graph_results, answer_data = self._graph.retrieve(cypher_result)
        graph_ids = {r.project_id for r in graph_results}

        # 3. Vector retrieval — semantic/fuzzy matching using LLM's vector_query
        # ────────── TEMPORARILY DISABLED VECTOR SEARCH ──────────
        # vector_results = self._vector.retrieve(
        #     intent,
        #     query_override=cypher_result.vector_query,
        # )
        vector_results = []
        # ────────────────────────────────────────────────────────

        # 4. Merge: graph results are canonical; vector fills in semantic gaps
        merged: dict[str, ProjectResult] = {}

        for r in graph_results:
            merged[r.project_id] = r

        for r in vector_results:
            if r.project_id not in merged:
                merged[r.project_id] = r
            else:
                merged[r.project_id].source = "both"
                merged[r.project_id].score = r.score

        # 5. Sort and filter
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

        # 6. Build direct answer text for LOOKUP / AGGREGATE
        direct_answer_text = ""
        if answer_data and cypher_result.query_type in ("LOOKUP", "AGGREGATE"):
            direct_answer_text = _format_answer_data(
                answer_data,
                query_type=cypher_result.query_type,
                raw_query=raw_query,
                params=cypher_result.params,
            )

        # 7. Assemble context text
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
        return context_text, final, direct_answer_text
