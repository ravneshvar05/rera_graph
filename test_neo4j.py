import sys
import os

# Add graphrag to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'graphrag')))

from neo4j import GraphDatabase
from graphrag_config import settings

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

def test():
    with driver.session() as session:
        # Check a specific project
        result = session.run('''
            MATCH (p:Project)
            WHERE p.project_name CONTAINS 'Neelkanth Status' OR p.project_name CONTAINS 'Sunflower Enclave' OR p.project_name CONTAINS 'DWARKESH GREENS'
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (p)-[:HAS_AMENITY]->(am:Amenity)
            RETURN p.project_name, count(DISTINCT u) as unit_count, collect(DISTINCT am.name) as amenities
        ''')
        for r in result:
            print(f"Project: {r['p.project_name']}, Units: {r['unit_count']}, Amenities: {r['amenities']}")
test()
driver.close()
