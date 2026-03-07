import time
import os
from dotenv import load_dotenv

# Load env before importing other modules
load_dotenv()

from query_parser import parse_query
from search import search
from answer_generator import generate_answer

def run_profile():
    query = "3 BHK Villa in Ahmedabad"
    
    print(f"Testing Query: '{query}'")
    print("-" * 40)
    
    # Test Parsing
    t0 = time.time()
    parsed = parse_query(query)
    t1 = time.time()
    print(f"[Parsing] Time taken: {t1 - t0:.2f} seconds")
    print(f"[Parsing] Extracted: city={parsed.city}, bhk={parsed.bhk}, type={parsed.property_type}")
    
    # Test Search
    t2 = time.time()
    results = search(parsed, top_k=5)
    t3 = time.time()
    print(f"[Search ] Time taken: {t3 - t2:.2f} seconds")
    print(f"[Search ] Results found: {len(results)}")
    
    # Test Generation
    t4 = time.time()
    answer = generate_answer(query, results)
    t5 = time.time()
    print(f"[Answer ] Time taken: {t5 - t4:.2f} seconds")
    print(f"[Answer ] Output length: {len(answer)} characters")
    
    print("-" * 40)
    print(f"Total Time: {t5 - t0:.2f} seconds")

if __name__ == "__main__":
    run_profile()
