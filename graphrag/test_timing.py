"""
test_timing.py — Measures per-step response times for the GraphRAG pipeline.

Run: cd d:\Rerachatbot\graphrag && .\venv\Scripts\python.exe test_timing.py
"""

import time
from graphrag_cypher import generate_cypher
from graphrag_retriever import DualRetriever

retriever = DualRetriever()

TEST_QUERIES = [
    "2 BHK in Vinzol with swimming pool",
    "show me all available projects",
    "luxury apartment near airport",
]

SEP = "=" * 60

total_times = []

for query in TEST_QUERIES:
    print(f"\n{SEP}")
    print(f"Query: {query!r}")

    # Step 1: LLM (Cypher + Intent)
    t0 = time.perf_counter()
    cypher_result = generate_cypher(query)
    t1 = time.perf_counter()
    llm_time = t1 - t0
    print(f"  [Step 1] LLM ({cypher_result.engine_used}):  {llm_time:.2f}s")

    # Step 2: Retrieval (graph + vector parallel)
    t2 = time.perf_counter()
    context_text, results, direct_answer = retriever.retrieve_and_assemble(
        cypher_result.intent, query, cypher_result=cypher_result
    )
    t3 = time.perf_counter()
    retrieval_time = t3 - t2
    total = t3 - t0
    total_times.append(total)

    print(f"  [Step 2] Retrieval:          {retrieval_time:.2f}s")
    print(f"  [Total]                      {total:.2f}s  →  {len(results)} projects found")
    print(f"  query_type={cypher_result.query_type}")

print(f"\n{SEP}")
avg = sum(total_times) / len(total_times)
print(f"Average total time: {avg:.2f}s  (across {len(TEST_QUERIES)} queries)")
print(f"Min: {min(total_times):.2f}s  |  Max: {max(total_times):.2f}s")
print(SEP)

retriever.close()
