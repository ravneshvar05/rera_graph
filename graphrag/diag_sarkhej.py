"""Quick diagnostic: check sarkhej/vinzol in Neo4j + test intent parsing."""
import os, sys
os.chdir(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

from neo4j import GraphDatabase

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(uri, auth=(user, pwd))

with driver.session() as s:
    # Check all neighbourhoods
    rows = list(s.run("MATCH (n:Neighbourhood) RETURN n.name AS name ORDER BY name"))
    print("=== NEIGHBOURHOODS IN DB ===")
    for row in rows:
        print(f"  {row['name']}")

    # Sarkhej/Vinzol projects
    rows2 = list(s.run(
        "MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood) "
        "WHERE toLower(n.name) CONTAINS 'sarkhej' "
        "   OR toLower(n.name) CONTAINS 'vinzol' "
        "   OR toLower(p.address) CONTAINS 'sarkhej' "
        "   OR toLower(p.address) CONTAINS 'vinzol' "
        "RETURN p.project_name AS pname, n.name AS nbhd, p.address AS addr "
        "LIMIT 20"
    ))
    print("\n=== SARKHEJ/VINZOL PROJECTS ===")
    if rows2:
        for r in rows2:
            print(f"  {r['pname']} | {r['nbhd']} | {r['addr']}")
    else:
        print("  (none found)")

    # Total count
    total = list(s.run("MATCH (p:Project) RETURN count(p) AS c"))[0]["c"]
    print(f"\nTotal projects in DB: {total}")

driver.close()

# Also test intent parsing
print("\n=== INTENT PARSE TEST ===")
try:
    from graphrag_intent import parse_intent
    intent = parse_intent("find me house in sarkhej and vinzol")
    print(f"  neighbourhood: {intent.neighbourhood}")
    print(f"  city: {intent.city}")
    print(f"  bhk: {intent.bhk}")
    print(f"  property_type: {intent.property_type}")
    print(f"  query_type: {intent.query_type}")
    print(f"  semantic_keywords: {intent.semantic_keywords}")
except Exception as e:
    print(f"  Intent parse error: {e}")
