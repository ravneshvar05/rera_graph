"""Check Zone nodes, PART_OF relationships, and what neighbourhoods map to which zones."""
import sys
sys.path.insert(0, ".")
from graphrag_config import settings
from neo4j import GraphDatabase

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

with driver.session() as session:
    # Check Zone nodes
    zones = list(session.run("MATCH (z:Zone) RETURN z.name AS zone ORDER BY zone"))
    print(f"=== Zone nodes ({len(zones)}) ===")
    for r in zones:
        print(f"  {r['zone']}")

    # Check PART_OF relationships: Neighbourhood -> Zone
    rels = list(session.run(
        "MATCH (n:Neighbourhood)-[:PART_OF]->(z:Zone) "
        "RETURN n.name AS neighbourhood, z.name AS zone ORDER BY z.name, n.name"
    ))
    print(f"\n=== Neighbourhood -[:PART_OF]-> Zone ({len(rels)}) ===")
    for r in rels:
        print(f"  {r['neighbourhood']} --> {r['zone']}")

    # If no PART_OF, check Neighbourhood nodes at all
    nbhs = list(session.run("MATCH (n:Neighbourhood) RETURN n.name AS name, n.zone AS zone ORDER BY name"))
    print(f"\n=== All Neighbourhood nodes ({len(nbhs)}) ===")
    for r in nbhs:
        print(f"  {r['name']} | zone_prop={r['zone']}")

driver.close()
