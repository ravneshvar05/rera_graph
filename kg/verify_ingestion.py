"""
Quick post-ingestion verification script.
Checks Neo4j node/relationship counts and ChromaDB embedding counts.
"""
from kg_config import settings
from neo4j import GraphDatabase
import chromadb

print("=== VERIFICATION AFTER RE-INGESTION ===\n")

# ── Neo4j ──────────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
)
with driver.session() as s:
    res = s.run(
        "MATCH (n) RETURN labels(n)[0] AS Label, count(n) AS Count ORDER BY Count DESC"
    )
    print("--- Neo4j Node Counts ---")
    node_map = {}
    for r in res:
        label = r["Label"]
        count = r["Count"]
        node_map[label] = count
        print(f"  {label}: {count}")

    res = s.run(
        "MATCH ()-[r]->() RETURN type(r) AS Type, count(r) AS Count ORDER BY Count DESC"
    )
    print("\n--- Neo4j Relationships ---")
    for r in res:
        print(f"  {r['Type']}: {r['Count']}")

driver.close()

# ── ChromaDB ───────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
col = client.get_collection(settings.CHROMA_COLLECTION_NAME)
total = col.count()

proj_ids = col.get(where={"doc_type": {"$eq": "project"}}, include=[])["ids"]
unit_ids = col.get(where={"doc_type": {"$eq": "unit"}}, include=[])["ids"]

print(f"\n--- ChromaDB ---")
print(f"  Collection  : {settings.CHROMA_COLLECTION_NAME}")
print(f"  Total docs  : {total}")
print(f"  Project docs: {len(proj_ids)}")
print(f"  Unit docs   : {len(unit_ids)}")

# ── Final verdict ─────────────────────────────────────────────────────────
projects_ok = node_map.get("Project", 0) > 0
chroma_ok   = total > 0
split_ok    = len(proj_ids) == node_map.get("Project", 0)

print("\n=== RESULT ===")
print(f"  Neo4j has projects     : {'YES' if projects_ok else 'NO !!!'}")
print(f"  ChromaDB has embeddings: {'YES' if chroma_ok else 'NO !!!'}")
print(f"  Project doc count match: {'YES' if split_ok else 'NO !!!'} "
      f"({len(proj_ids)} chroma == {node_map.get('Project', 0)} neo4j)")

if projects_ok and chroma_ok and split_ok:
    print("\n  ALL CHECKS PASSED — system is intact.")
else:
    print("\n  !!! SOMETHING IS WRONG — check above.")
