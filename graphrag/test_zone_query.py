"""Test zone filter Cypher directly (bypassing LLM intent parsing)."""
import sys
sys.path.insert(0, ".")
from graphrag_config import settings
from graphrag_intent import QueryIntent
from graphrag_retriever import GraphRetriever

# Manually set intent - bypass LLM
intent = QueryIntent(
    query_type="SPECIFIC",
    zone="West Ahmedabad",
    city=None,
    neighbourhood=None,
)
print(f"Intent: zone={intent.zone!r}\n")

retriever = GraphRetriever()
results = retriever._get_filtered_projects(intent)
print(f"Projects in West Ahmedabad: {len(results)}")
for r in results:
    print(f"  - {r.project_name} | {r.neighbourhood}")

print()

# Also test east
intent2 = QueryIntent(query_type="SPECIFIC", zone="East Ahmedabad")
results2 = retriever._get_filtered_projects(intent2)
print(f"Projects in East Ahmedabad: {len(results2)}")
for r in results2:
    print(f"  - {r.project_name} | {r.neighbourhood}")

retriever.close()
