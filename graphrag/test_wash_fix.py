"""Test: does the new ANY(r IN u.rooms) pattern return the expected projects?"""
from neo4j import GraphDatabase
import warnings; warnings.filterwarnings("ignore")

driver = GraphDatabase.driver(
    "neo4j+ssc://d00bd26c.databases.neo4j.io",
    auth=("d00bd26c", "s5QXVdHqSkWT1AUgj72jhYEWeen-SxErQmggqC1KSxY")
)

q = """
MATCH (p:Project)-[:LOCATED_IN]->(n:Neighbourhood)-[:IN_CITY]->(c:City)
OPTIONAL MATCH (p)-[:BUILT_BY]->(dev:Developer)
CALL {
  WITH p
  OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
  WITH u WHERE u IS NOT NULL
  RETURN collect(u { .*, rooms: [(u)-[:HAS_ROOM]->(r:Room) | properties(r)] }) AS units
}
WITH p, n, c, dev, units
WHERE ANY(u IN units WHERE
  u.bhk = 2
  AND ANY(r IN u.rooms WHERE r.name IN ['Wash Area'])
  AND (
    (u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0)
    OR ANY(r IN u.rooms WHERE r.name IN ['Balcony', 'Terrace'])
  )
)
RETURN p.project_name AS project ORDER BY project
"""

with driver.session() as s:
    rows = list(s.run(q))

print(f"\nFound {len(rows)} project(s) matching '2 BHK with wash area AND balcony':")
for r in rows:
    print(f"  - {r['project']}")

expected = {"svasaar pravesh", "OUM ORBIT"}
found = {r["project"] for r in rows}
missing = expected - found
extra = found - expected

print(f"\nExpected: {sorted(expected)}")
print(f"Missing:  {sorted(missing) or 'None'}")
print(f"Extra:    {sorted(extra) or 'None'}")
print(f"\nResult: {'PASS' if not missing else 'FAIL - missing expected projects'}")

driver.close()
