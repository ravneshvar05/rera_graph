import asyncio
import logging
import json
from pathlib import Path
from graphrag_cypher import generate_cypher
from graphrag_retriever import DualRetriever, GraphRetriever

logging.basicConfig(level=logging.WARNING)

def test_query(query_text):
    print(f"\n==================================================")
    print(f"Testing query: '{query_text}'")
    
    user_settings = {}
    settings_path = Path("user_settings.json")
    if settings_path.exists():
        with open(settings_path, "r") as f:
            user_settings = json.load(f)
            
    cypher_result = generate_cypher(query_text, api_keys=user_settings)
    intent = cypher_result.intent
    
    print(f"\nParsed Intent:")
    print(intent.model_dump_json(indent=2))
    
    print(f"\nGenerated Cypher:")
    print(cypher_result.cypher)
    print(f"Params: {cypher_result.params}")
    
    retriever = DualRetriever()
    context_text, results, direct_answer_text = retriever.retrieve_and_assemble(
        intent, query_text, cypher_result=cypher_result
    )
    
    print(f"\nTotal Projects Found: {len(results)}")
    
    for r in results:
        print(f" - {r.project_name} (Source: {r.source})")
        
    print("==================================================\n")

if __name__ == "__main__":
    test_query("Find me projects in Ahmedabad that do not have commercial shops included but have a clubhouse and park.")
