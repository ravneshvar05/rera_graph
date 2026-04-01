"""Full pipeline diagnostic for 'find me house in sarkhej and vinzol'."""
import os
os.chdir(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

TEST_QUERY = "find me house in sarkhej and vinzol"

print("=" * 60)
print(f"TEST QUERY: {TEST_QUERY}")
print("=" * 60)

# Step 1: Test generate_cypher
print("\n--- Step 1: generate_cypher ---")
try:
    from graphrag_cypher import generate_cypher
    cq = generate_cypher(TEST_QUERY)
    print(f"  Engine: {cq.engine_used}")
    print(f"  Query type: {cq.query_type}")
    print(f"  Intent neighbourhood: {cq.intent.neighbourhood}")
    print(f"  Intent city: {cq.intent.city}")
    print(f"  Intent bhk: {cq.intent.bhk}")
    print(f"  Intent semantic_keywords: {cq.intent.semantic_keywords}")
    print(f"  Cypher (first 300 chars): {cq.cypher[:300]}")
    print(f"  Params: {cq.params}")
    print(f"  Vector query: {cq.vector_query}")
except Exception as e:
    print(f"  ERROR: {e}")
    cq = None

# Step 2: Test graph retriever
print("\n--- Step 2: Graph retriever ---")
try:
    from graphrag_retriever import GraphRetriever
    if cq:
        gr = GraphRetriever()
        results, answer_data = gr.retrieve(cq)
        print(f"  Graph results: {len(results)}")
        for r in results:
            print(f"    - {r.project_name} | {r.neighbourhood}, {r.city} | source={r.source}")
        gr.close()
    else:
        print("  Skipped (no cypher)")
except Exception as e:
    print(f"  ERROR: {e}")

# Step 3: Test vector retriever
print("\n--- Step 3: Vector retriever ---")
try:
    from graphrag_retriever import VectorRetriever
    if cq:
        vr = VectorRetriever()
        vresults = vr.retrieve(cq.intent, query_override=cq.vector_query)
        print(f"  Vector results: {len(vresults)}")
        for r in vresults[:5]:
            print(f"    - {r.project_name} | {r.neighbourhood}, {r.city} | score={r.score:.3f}")
    else:
        print("  Skipped (no cypher)")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- DONE ---")
