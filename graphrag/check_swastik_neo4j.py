"""
Diagnostic: check what neighbourhood name and amenities Swastik Harmony has in Neo4j.
"""
from neo4j import GraphDatabase
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from graphrag_config import settings

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

with driver.session() as session:
    # 1. Check Swastik Harmony's neighbourhood in graph
    print("=== Swastik Harmony in Neo4j ===")
    r = session.run("""
        MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
        WHERE toLower(p.project_name) CONTAINS 'swastik'
        OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
        RETURN p.project_name AS name, n.name AS neighbourhood, c.name AS city,
               collect(DISTINCT am.name) AS amenity_names,
               collect(DISTINCT am.canonical_tags) AS amenity_tags
    """)
    for rec in r:
        print(f"  Project: {rec['name']}")
        print(f"  Neighbourhood: '{rec['neighbourhood']}'")
        print(f"  City: {rec['city']}")
        print(f"  Amenity Names: {rec['amenity_names']}")
        print(f"  Canonical Tags: {rec['amenity_tags']}")
    
    print("\n=== Test multi-location + gym query ===")
    r2 = session.run("""
        MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
        OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
        OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
        OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
        WITH p, n, c, dev,
             collect(DISTINCT u) AS units,
             collect(DISTINCT am.name) AS amenities
        WHERE (
            toLower(n.name) CONTAINS toLower('Shantigram') OR toLower(p.address) CONTAINS toLower('Shantigram') OR
            toLower(n.name) CONTAINS toLower('Satellite') OR toLower(p.address) CONTAINS toLower('Satellite') OR
            toLower(n.name) CONTAINS toLower('New Nikol-Naroda Road') OR toLower(p.address) CONTAINS toLower('New Nikol-Naroda Road') OR
            toLower(n.name) CONTAINS toLower('New Nikol - Naroda Road') OR toLower(p.address) CONTAINS toLower('New Nikol - Naroda Road')
        )
        AND ANY(am IN amenities WHERE toLower(am) CONTAINS 'gym')
        RETURN p.project_name AS name, n.name AS neighbourhood, amenities
    """)
    print("  Projects with gym in those 3 areas:")
    for rec in r2:
        print(f"  - {rec['name']} ({rec['neighbourhood']}) | amenities: {rec['amenities']}")
    
    print("\n=== Check canonical_tags for Gym in Swastik Harmony ===")
    r3 = session.run("""
        MATCH (p:Project)-[:HAS_AMENITY]->(am:Amenity)
        WHERE toLower(p.project_name) CONTAINS 'swastik'
        RETURN am.name AS amenity, am.canonical_tags AS tags
    """)
    for rec in r3:
        print(f"  Amenity: '{rec['amenity']}' | Tags: {rec['tags']}")
    
    print("\n=== All neighbourhood names in DB ===")
    r4 = session.run("MATCH (n:Neighbourhood) RETURN n.name AS nb ORDER BY nb")
    for rec in r4:
        print(f"  '{rec['nb']}'")

driver.close()
