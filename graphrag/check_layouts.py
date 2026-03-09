import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
AUTH = (os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'testpassword'))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        result = session.run('MATCH (p:Project)-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout) RETURN p.project_name as project, f.layout_name as name, f.total_units_on_floor as units LIMIT 10')
        for r in result:
            print(f"Project: {r['project']}, Layout: {r['name']}, Units/Floor: {r['units']}")
