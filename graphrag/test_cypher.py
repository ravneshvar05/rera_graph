import json
from neo4j import GraphDatabase
from graphrag_config import settings

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
cypher = """
MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)
WITH p, collect(DISTINCT u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }) AS units
RETURN units LIMIT 1
"""
with driver.session() as session:
    res = session.run(cypher).data()

with open('output.json', 'w') as f:
    json.dump(res, f, indent=2)
