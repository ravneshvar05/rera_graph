"""
kg_schema.py — Create Neo4j uniqueness constraints and performance indexes.

Run once before first ingestion:
    python kg_schema.py

Safe to re-run — all constraints and indexes use IF NOT EXISTS.
"""

import sys
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from kg_config import settings

# ─── Constraint definitions ──────────────────────────────────────────────────

CONSTRAINTS = [
    # Node uniqueness constraints
    ("project_unique",     "Project",      "project_id"),
    ("city_unique",        "City",         "name"),
    ("neighbourhood_unique","Neighbourhood","name"),
    ("zone_unique",        "Zone",         "name"),
    ("developer_unique",   "Developer",    "name"),
    ("unit_unique",        "Unit",         "unit_id"),
    ("floorlayout_unique", "FloorLayout",  "layout_id"),
    ("amenity_unique",     "Amenity",      "name"),
    ("landmark_unique",    "Landmark",     "name"),
]

# ─── Index definitions ───────────────────────────────────────────────────────

INDEXES = [
    # Single-property indexes
    ("idx_unit_bhk",         "Unit",     ["bhk"]),
    ("idx_unit_ptype",       "Unit",     ["property_type"]),
    ("idx_unit_sqft",        "Unit",     ["super_builtup_sqft"]),
    ("idx_landmark_type",    "Landmark", ["landmark_type"]),
    ("idx_amenity_category", "Amenity",  ["category"]),
    ("idx_project_status",   "Project",  ["project_status"]),

    # Composite indexes for the most common traversal patterns
    ("idx_unit_bhk_ptype",   "Unit",     ["bhk", "property_type"]),
]


def create_constraints(session) -> None:
    for name, label, prop in CONSTRAINTS:
        cypher = (
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        try:
            session.run(cypher)
            logger.success(f"  ✓ Constraint: {name}  ({label}.{prop})")
        except Neo4jError as e:
            logger.error(f"  ✗ Constraint {name}: {e.message}")


def create_indexes(session) -> None:
    for name, label, props in INDEXES:
        prop_str = ", ".join(f"n.{p}" for p in props)
        cypher = (
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON ({prop_str})"
        )
        try:
            session.run(cypher)
            logger.success(f"  ✓ Index: {name}  ({label} → {props})")
        except Neo4jError as e:
            logger.error(f"  ✗ Index {name}: {e.message}")


def run_schema() -> None:
    logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI} ...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        logger.info("Connected.")

        with driver.session() as session:
            logger.info("Creating constraints ...")
            create_constraints(session)

            logger.info("Creating indexes ...")
            create_indexes(session)

        logger.info("Schema setup complete.")

    except Exception as e:
        logger.error(f"Failed to connect or apply schema: {e}")
        sys.exit(1)

    finally:
        driver.close()


if __name__ == "__main__":
    run_schema()
