"""
kg_geo_enrich.py — Post-ingestion geographic enrichment.

Run ONCE after the initial kg_ingest.py run.

What it does:
  1. Creates Zone nodes and wires Neighbourhoods → Zones → Cities
  2. Cross-links Landmark nodes to Neighbourhood nodes (SAME_AS edges)
     e.g. "BOPAL" landmark → Bopal neighbourhood node
  3. Wires neighbourhood alias edges (ALIAS_OF)
     e.g. Vizol → Vinzol, Gamdi → Gamdi Gaam

After this runs, these Cypher queries become possible:
  MATCH (p)-[:LOCATED_IN]->(:Neighbourhood)-[:PART_OF]->(:Zone {name: "West Ahmedabad"})
  MATCH (p)-[:NEAR]->(lm)-[:SAME_AS]->(n:Neighbourhood {name: "Bopal"})
  MATCH (n:Neighbourhood)-[:ALIAS_OF*1..2]->(canonical) WHERE canonical.name = "Vinzol"

Usage:
    python kg_geo_enrich.py          # run enrichment
    python kg_geo_enrich.py --reset  # wipe PART_OF / SAME_AS / ALIAS_OF edges and re-run
"""

import argparse
import sys
from loguru import logger
from neo4j import GraphDatabase

from kg_config import settings

# ══════════════════════════════════════════════════════════════════════════════
#  ZONE → NEIGHBOURHOOD MAPPINGS
#  Extend these dicts as new cities / neighbourhoods are added.
#  No code changes needed — just update the data here.
# ══════════════════════════════════════════════════════════════════════════════

CITY_ZONES: dict[str, dict[str, list[str]]] = {
    "Ahmedabad": {
        "West Ahmedabad": [
            # Canonical names
            "Bopal", "South Bopal", "New Bopal", "Ambli",
            "Satellite", "Prahlad Nagar", "Anand Nagar", "Vastrapur",
            "Bodakdev", "Thaltej", "Gokuldham", "Shilaj", "Science City Road",
            # Exact DB neighbourhood names
            "SHELA", "Shela",
            "Thaltej - Shilaj Road",
            "Iscon-Ambli Road",
            "Shantigram",
            "Jodhpur, Satellite",
            "Jodhpur Satellite",
            "Sola",
            "Shantipura Cross Roads, Bopal, Iscon-Ambli Road, Sardar Patel Ring Road",
        ],
        "East Ahmedabad": [
            "Nikol", "New Nikol", "Naroda", "New Naroda", "Vastral",
            "Kathwada", "Odhav", "Vastral Road",
            # Exact DB neighbourhood names
            "New Nikol - Naroda Road",
            "Naroda, Hanspura Road",
            "Ramol Village",
            "New Vatva",
            "Hathijan",
            "Kubernagar",
        ],
        "North Ahmedabad": [
            "Chandkheda", "Ranip", "New Ranip", "Gota", "Motera",
            "Chiloda", "Kotarpur", "Chiloda-Kotarpur",
            # Exact DB neighbourhood names
            "Manipur",
        ],
        "South Ahmedabad": [
            "Sarkhej", "Vejalpur", "Juhapura", "Narol", "Vatva",
            "Isanpur", "Gamdi Gaam", "Gamdi", "Vinzol", "Vizol",
            "Hanspur", "Hanspura",
            # Exact DB neighbourhood names
            "Vatva", "New Vatva",
            "Danteshwar",
            "Vishala Circle",
            "Vardhman Nagar",
        ],
        "Central Ahmedabad": [
            "Maninagar", "Paldi", "Navrangpura", "Ellisbridge", "Ghatlodia",
            # Exact DB neighbourhood names
            "NARANPURA", "Naranpura",
            "Race Course",
        ],
    },
    "Surat": {
        "West Surat":  [
            "Adajan", "Pal", "Vesu", "Piplod",
            # Exact DB neighbourhood names
            "Vesu, Udhana Magdalla Road",
        ],
        "East Surat":  ["Sarthana", "Katargam", "Varachha"],
        "North Surat": [
            "Jahangirpura",
        ],
    },
    "Vadodara": {
        "Vadodara": [
            "Vasna-Bhayli Road, Bhyali", "Vasna-Bhayli Road", "Bhyali",
        ],
    },
}

# ── Landmark name → Neighbourhood name cross-links ────────────────────────────
# If a project is near this landmark, it often means it's near that neighbourhood.
LANDMARK_TO_NEIGHBOURHOOD: dict[str, str] = {
    # Ahmedabad
    "BOPAL":         "Bopal",
    "SATELLITE":     "Satellite",
    "BODAKDEV":      "Bodakdev",
    "SARKHEJ":       "Sarkhej",
    "NIKOL":         "Nikol",
    "NARODA":        "Naroda",
    "THALTEJ":       "Thaltej",
    "PRAHLAD NAGAR": "Prahlad Nagar",
    "VASTRAPUR":     "Vastrapur",
    "SHELA":         "Shela",
    "ISANPUR":       "Isanpur",
    "VATVA":         "Vatva",
    "GOTA":          "Gota",
    "VASTRAL":       "Vastral",
    # Surat
    "ADAJAN":        "Adajan",
    "VESU":          "Vesu",
    "KATARGAM":      "Katargam",
}

# ── Neighbourhood aliases: variant → canonical ────────────────────────────────
NEIGHBOURHOOD_ALIASES: dict[str, str] = {
    # Ahmedabad spellings / abbreviations
    "Vizol":      "Vinzol",
    "Vejalpur":   "Vejalpur",    # also "Vejalpore" in some brochures
    "Vejalpore":  "Vejalpur",
    "Gamdi":      "Gamdi Gaam",
    "New Nikol":  "Nikol",
    "New Naroda": "Naroda",
    "New Bopal":  "Bopal",
    "Science City": "Science City Road",
    "SG Highway": "Sarkhej",
    "SG Road":    "Sarkhej",
    # Chiloda variants
    "Chiloda Kotarpur": "Chiloda-Kotarpur",
    "Nana Chiloda":     "Chiloda-Kotarpur",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CYPHER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CYPHER_ZONE_LINK = """
MERGE (city:City {name: $city_name})
MERGE (zone:Zone {name: $zone_name})
MERGE (zone)-[:IN_CITY]->(city)
MERGE (hood:Neighbourhood {name: $hood_name})
MERGE (hood)-[:IN_CITY]->(city)
MERGE (hood)-[:PART_OF]->(zone)
"""

_CYPHER_LANDMARK_CROSSLINK = """
MATCH (lm:Landmark)
WHERE toLower(lm.name) CONTAINS toLower($lm_keyword)
MATCH (n:Neighbourhood {name: $hood_name})
MERGE (lm)-[:SAME_AS]->(n)
"""

_CYPHER_ALIAS = """
MERGE (alias:Neighbourhood {name: $alias_name})
MERGE (canonical:Neighbourhood {name: $canonical_name})
MERGE (alias)-[:ALIAS_OF]->(canonical)
"""

_CYPHER_RESET = """
MATCH ()-[r:PART_OF|SAME_AS|ALIAS_OF]->()
DELETE r
"""

_CYPHER_DELETE_ZONES = """
MATCH (z:Zone) DETACH DELETE z
"""


def run_geo_enrich(reset: bool = False) -> None:
    logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI} ...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        logger.success("Connected.")

        with driver.session() as session:

            if reset:
                logger.warning("--reset: deleting PART_OF / SAME_AS / ALIAS_OF edges + Zone nodes ...")
                session.run(_CYPHER_RESET)
                session.run(_CYPHER_DELETE_ZONES)
                logger.warning("Reset complete.")

            # ── 1. Zone hierarchy ─────────────────────────────────────────────
            total_zone_links = 0
            logger.info("Wiring Zone hierarchy ...")
            for city_name, zones in CITY_ZONES.items():
                for zone_name, hoods in zones.items():
                    for hood_name in hoods:
                        session.run(_CYPHER_ZONE_LINK, {
                            "city_name": city_name,
                            "zone_name": zone_name,
                            "hood_name": hood_name,
                        })
                        total_zone_links += 1
            logger.success(f"  ✓ {total_zone_links} zone–neighbourhood links created.")

            # ── 2. Landmark → Neighbourhood cross-links ────────────────────────
            logger.info("Cross-linking landmarks → neighbourhoods (SAME_AS) ...")
            matched_lm = 0
            for lm_keyword, hood_name in LANDMARK_TO_NEIGHBOURHOOD.items():
                result = session.run(_CYPHER_LANDMARK_CROSSLINK, {
                    "lm_keyword": lm_keyword,
                    "hood_name":  hood_name,
                })
                summary = result.consume()
                rels_created = summary.counters.relationships_created
                if rels_created > 0:
                    logger.debug(f"  {lm_keyword!r} → {hood_name}  ({rels_created} link(s))")
                    matched_lm += rels_created
            logger.success(f"  ✓ {matched_lm} landmark→neighbourhood SAME_AS edge(s) created.")

            # ── 3. Neighbourhood aliases ───────────────────────────────────────
            logger.info("Wiring neighbourhood aliases (ALIAS_OF) ...")
            alias_count = 0
            for alias, canonical in NEIGHBOURHOOD_ALIASES.items():
                session.run(_CYPHER_ALIAS, {
                    "alias_name":    alias,
                    "canonical_name": canonical,
                })
                alias_count += 1
            logger.success(f"  ✓ {alias_count} neighbourhood alias ALIAS_OF edge(s) created.")

        logger.success("Geo enrichment complete.")

    except Exception as e:
        logger.error(f"Geo enrichment failed: {e}")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    parser = argparse.ArgumentParser(description="Wire Zone hierarchy + landmark aliases in Neo4j")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing PART_OF / SAME_AS / ALIAS_OF edges and re-run",
    )
    args = parser.parse_args()
    run_geo_enrich(reset=args.reset)
