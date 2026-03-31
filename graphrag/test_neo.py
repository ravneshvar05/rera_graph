from neo4j import GraphDatabase
from graphrag_config import settings

def test_neo4j():
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        records = session.run("MATCH (p:Project) RETURN p.project_name, p.has_commercial_shops, p.has_clubhouse, p.has_park")
        for r in records:
            if r['p.project_name'] in ['OUM ORBIT', 'Swastik Harmony', 'KP Villas']:
                print(r['p.project_name'], ":", "shops:", r['p.has_commercial_shops']," | club:", r['p.has_clubhouse']," | park:", r['p.has_park'])

if __name__ == "__main__":
    test_neo4j()
