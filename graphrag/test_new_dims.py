import os
import sys
from dotenv import load_dotenv

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from graphrag_cypher import generate_cypher

tests = [
    "2 BHK flat with bedroom around 10 by 12",
    "project with master bedroom at least 15 by 12",
    "villa with kitchen less than 8x10",
    "apartment with hall exactly 10 by 10",
]

for query in tests:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    
    cq = generate_cypher(query)
    
    print("\n--- EXTRACTED INTENT ---")
    if cq.intent and cq.intent.room_dimensions:
        for i, rd in enumerate(cq.intent.room_dimensions):
            print(f"  Room {i+1}: {rd.get('room')} -> {rd.get('d1')} x {rd.get('d2')}")
            print(f"  Qualifier: {rd.get('dim_qualifier')}")
    else:
        print("  None found!")
        
    print("\n--- GENERATED CYPHER (Snippet) ---")
    lines = cq.cypher.split("\n")
    # Only show the relevant WHERE EXISTS conditions for the dimensions
    for line in lines:
        if "EXISTS" in line or "length_ft" in line or "width_ft" in line:
            print("  " + line.strip())
            
    print("\n--- PARAMS ---")
    relevant_params = {k: v for k, v in cq.params.items() if 'dim' in k or 'd1' in k or 'd2' in k or 'room_names' in k}
    print(f"  {relevant_params}")
    print()
