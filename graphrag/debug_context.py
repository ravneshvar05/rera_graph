import json
from graphrag_intent import parse_intent
from graphrag_retriever import DualRetriever

q = "how many units per floor in oum orbit"
print(f"User Query: {q}")

intent = parse_intent(q)
print(f"Parsed Intent: {intent.model_dump_json(indent=2)}")

d = DualRetriever()
ctx, projs = d.retrieve_and_assemble(intent, q)

print("\n--- ASSEMBLED CONTEXT FOR LLM ---\n")
print(ctx)
print("\n---------------------------------\n")

print(f"Total projects retrieved: {len(projs)}")
if projs:
    p = projs[0]
    print(f"First Project: {p.project_name}")
    print(f"Floor Layouts Raw: {getattr(p, 'floor_layouts', 'NOT PRESENT')}")
