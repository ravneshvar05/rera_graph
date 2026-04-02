"""
Verify length_ft / width_ft fields are correctly stored in Neo4j Room nodes.
Run from: d:\Rerachatbot\kg\
With:     d:\Rerachatbot\kg\venv\Scripts\python.exe tmp_verify_dims_kg.py
"""
from kg_config import settings
from neo4j import GraphDatabase

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

with driver.session() as s:

    # 1. Count rooms and how many have length_ft set
    row = s.run("""
        MATCH (r:Room)
        RETURN count(r) AS total,
               count(r.length_ft) AS has_lft,
               count(r.width_ft)  AS has_wft
    """).single()
    print("=== Room Node Counts ===")
    print(f"  Total rooms       : {row['total']}")
    print(f"  With length_ft    : {row['has_lft']}")
    print(f"  With width_ft     : {row['has_wft']}")

    # 2. Sample rows
    print("\n=== Sample: raw length -> length_ft ===")
    rows = s.run("""
        MATCH (r:Room)
        WHERE r.length IS NOT NULL AND r.length_ft IS NOT NULL
        RETURN r.name AS name, r.length AS raw_l, r.length_ft AS lft,
               r.width AS raw_w, r.width_ft AS wft
        LIMIT 15
    """)
    for r in rows:
        print(f"  {r['name']:<18} {str(r['raw_l']):<14} -> {r['lft']:<8}  |  {str(r['raw_w']):<14} -> {r['wft']}")

    # 3. Test 10x10 match (KP Villas has Courtyard, Garden, Terrace 10x10)
    print("\n=== Query: rooms with length_ft=10.0, width_ft=10.0 ===")
    rows2 = s.run("""
        MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
        WHERE r.length_ft IS NOT NULL AND r.width_ft IS NOT NULL
        AND toFloat(r.length_ft) = 10.0 AND toFloat(r.width_ft) = 10.0
        RETURN p.project_name AS proj, u.unit_type AS unit, r.name AS room,
               r.length AS raw_l, r.width AS raw_w
        LIMIT 10
    """)
    found = False
    for r in rows2:
        found = True
        print(f"  {r['proj']:<28} | {r['unit']:<14} | {r['room']:<18} | {r['raw_l']} x {r['raw_w']}")
    if not found:
        print("  NO RESULTS (unexpected - KP Villas has 10x10 Courtyard/Garden/Terrace)")

    # 4. Test 10x12 orientation-agnostic (KP Villas has Hall, Bedroom 10'x12')
    print("\n=== Query: rooms 10x12 (orientation-agnostic) ===")
    rows3 = s.run("""
        MATCH (p:Project)-[:HAS_UNIT]->(u:Unit)-[:HAS_ROOM]->(r:Room)
        WHERE r.length_ft IS NOT NULL AND r.width_ft IS NOT NULL
        AND (
          (toFloat(r.length_ft) = 10.0 AND toFloat(r.width_ft) = 12.0)
          OR (toFloat(r.length_ft) = 12.0 AND toFloat(r.width_ft) = 10.0)
        )
        RETURN p.project_name AS proj, u.unit_type AS unit, r.name AS room,
               r.length AS raw_l, r.width AS raw_w
        LIMIT 10
    """)
    found2 = False
    for r in rows3:
        found2 = True
        print(f"  {r['proj']:<28} | {r['unit']:<14} | {r['room']:<18} | {r['raw_l']} x {r['raw_w']}")
    if not found2:
        print("  NO RESULTS (unexpected - KP Villas has Hall/Bedroom 10x12)")

driver.close()
print("\nDone.")
