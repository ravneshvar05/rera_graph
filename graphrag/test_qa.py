import json
from graphrag_intent import parse_intent
from graphrag_retriever import DualRetriever

queries = [
    "flat with hall > 150 sqft in ahmedabad",
    "give me the hall size of 2 bhk projects in ahmedabad"
]

d = DualRetriever()
results = []
for q in queries:
    intent = parse_intent(q)
    ctx, projs = d.retrieve_and_assemble(intent, q)
    results.append({
        "query": q,
        "context": ctx,
        "parsed_intent": intent.model_dump()
    })

with open("context.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

d.close()
