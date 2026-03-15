from graphrag_retriever import GraphRetriever
from graphrag_cypher import generate_cypher

retriever = GraphRetriever()
q = "What is the hall size in OUM Orbit Ahmedabad?"
print("Generating Cypher...")
cq = generate_cypher(q)
print(f"Cypher:\n{cq.cypher}\nParams: {cq.params}")

print("\nExecuting against Neo4j...")
results, answer_data = retriever._execute_cypher(cq)
print(f"Projects found: {len(results)}")
print(f"Answer data items: {len(answer_data)}")
if results:
    print(f"First project: {results[0].project_name}")

retriever.close()
