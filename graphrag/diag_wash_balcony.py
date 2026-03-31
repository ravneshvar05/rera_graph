"""
Diagnostic: Check which projects have 2 BHK units with Wash Area room + balcony.
"""
from neo4j import GraphDatabase

URI  = "neo4j+ssc://d00bd26c.databases.neo4j.io"
USER = "d00bd26c"
PASS = "s5QXVdHqSkWT1AUgj72jhYEWeen-SxErQmggqC1KSxY"

driver = GraphDatabase.driver(URI, auth=(USER, PASS))

print("=" * 70)
print("1. Projects with 2 BHK units that have a 'Wash Area' room")
print("=" * 70)
q1 = """
MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
WHERE u.bhk = 2 AND toLower(r.name) CONTAINS 'wash'
RETURN DISTINCT p.project_name AS project, u.unit_id AS unit_id,
       u.balcony_sqft AS balcony_sqft, r.name AS room_name
ORDER BY project
"""
with driver.session() as s:
    for rec in s.run(q1):
        print(f"  {rec['project']:30s} | unit={rec['unit_id']} | room={rec['room_name']} | balcony={rec['balcony_sqft']}")

print()
print("=" * 70)
print("2. Projects with 2 BHK units that have balcony_sqft > 0")
print("=" * 70)
q2 = """
MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)
WHERE u.bhk = 2 AND u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0
RETURN DISTINCT p.project_name AS project, u.unit_id AS unit_id,
       u.balcony_sqft AS balcony_sqft
ORDER BY project
"""
with driver.session() as s:
    for rec in s.run(q2):
        print(f"  {rec['project']:30s} | unit={rec['unit_id']} | balcony={rec['balcony_sqft']}")

print()
print("=" * 70)
print("3. All Room names for svasaar pravesh & OUM ORBIT (2 BHK)")
print("=" * 70)
q3 = """
MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
WHERE u.bhk = 2 AND toLower(p.project_name) IN ['svasaar pravesh','oum orbit']
RETURN p.project_name AS project, u.unit_id AS unit_id,
       collect(DISTINCT r.name) AS rooms, u.balcony_sqft AS balcony_sqft
ORDER BY project, unit_id
"""
with driver.session() as s:
    for rec in s.run(q3):
        print(f"  {rec['project']:20s} | unit={rec['unit_id']} | balcony={rec['balcony_sqft']}")
        print(f"    Rooms: {rec['rooms']}")

print()
print("=" * 70)
print("4. Correlated query: 2 BHK with SAME unit having Wash Area + balcony")
print("   (Using unit_id correlation)")
print("=" * 70)
q4 = """
MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
WHERE u.bhk = 2
  AND toLower(r.name) CONTAINS 'wash'
  AND u.balcony_sqft IS NOT NULL AND toFloat(u.balcony_sqft) > 0
RETURN DISTINCT p.project_name AS project, u.unit_id AS unit_id,
       u.balcony_sqft AS balcony, r.name AS wash_room
ORDER BY project
"""
with driver.session() as s:
    rows = list(s.run(q4))
    if rows:
        for rec in rows:
            print(f"  ✓ {rec['project']:30s} | unit={rec['unit_id']} | wash={rec['wash_room']} | balcony={rec['balcony']}")
    else:
        print("  ✗ NO results — means no single unit satisfies BOTH conditions together")

print()
print("=" * 70)
print("5. Relaxed: projects with EITHER wash area (any unit) OR balcony>0 (any 2BHK unit)")
print("=" * 70)
q5 = """
MATCH (p:Project)
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u1:Unit)-[:HAS_ROOM]->(r1:Room)
  WHERE u1.bhk = 2 AND toLower(r1.name) CONTAINS 'wash'
OPTIONAL MATCH (p)-[:HAS_UNIT]->(u2:Unit)
  WHERE u2.bhk = 2 AND u2.balcony_sqft IS NOT NULL AND toFloat(u2.balcony_sqft) > 0
WITH p, count(DISTINCT u1) AS wash_count, count(DISTINCT u2) AS balcony_count
WHERE wash_count > 0 OR balcony_count > 0
RETURN p.project_name AS project, wash_count, balcony_count
ORDER BY project
"""
with driver.session() as s:
    for rec in s.run(q5):
        print(f"  {rec['project']:30s} | wash_units={rec['wash_count']} | balcony_units={rec['balcony_count']}")

driver.close()
print("\nDone.")
