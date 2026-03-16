"""
Test script to verify graph retrieval fixes for test questions Q1-Q12.

Runs generate_cypher() for each failing test question and checks if:
1. The generated Cypher contains p.address search
2. The Cypher executes successfully against Neo4j
3. The ground truth project is in the results
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from graphrag_cypher import generate_cypher
from neo4j import GraphDatabase
from graphrag_config import settings

driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

# Test cases: (question, expected_project_names, description)
TEST_CASES = [
    (
        "Can you suggest some 2 BHK apartments in Sarkhej, Vinzol, or Gamdi Gaam?",
        ["svasaar pravesh", "Sunflower Enclave", "AMBER", "OUM ORBIT"],
        "Q1: Multi-location including Gamdi Gaam (only in address)"
    ),
    (
        "Are there any apartments with a gym in Shantigram, Satellite, or New Nikol-Naroda Road?",
        ["Swastik Harmony", "Elysium"],
        "Q4: New Nikol (compound neighbourhood name) + gym amenity"
    ),
    (
        "Show me properties near Kotarpur Water Works that offer 4 BHK villas.",
        ["Sudama Homes"],
        "Q7: Landmark-based search (Kotarpur Water Works)"
    ),
    (
        "Is there a 2 BHK project located opposite the Karnavati APMC Market?",
        ["OUM ORBIT"],
        "Q8: Landmark-based search (Karnavati APMC Market)"
    ),
    (
        "Can you find me an apartment situated on Shaikh Adam Abuvala Road in Paldi?",
        ["Gandhi Galaxy"],
        "Q10: Road name only in address, Paldi in neighbourhood"
    ),
    (
        "I'm searching for a property near the 200 FT. Ring Road in the Hanspura area.",
        ["Devnandan Sankalp"],
        "Q11: 200 FT Ring Road (landmark) + Hanspura (in neighbourhood name)"
    ),
]


def run_test(question, expected_names, description):
    """Run a single test case and return (pass, details) tuple."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"QUERY: {question}")
    
    try:
        cq = generate_cypher(question)
    except Exception as e:
        print(f"  FAIL: generate_cypher() raised {e}")
        return False, str(e)
    
    print(f"  Type:    {cq.query_type}")
    print(f"  Params:  {cq.params}")
    print(f"  Vector:  {cq.vector_query}")
    
    # Check if p.address is referenced in the Cypher
    has_address = "p.address" in cq.cypher
    print(f"  Has p.address search: {'YES' if has_address else 'NO'}")
    
    # Check intent
    intent = cq.intent
    print(f"  Intent neighbourhood: {intent.neighbourhood}")
    print(f"  Intent landmarks: {intent.specific_landmarks}")
    
    # Execute the Cypher
    try:
        with driver.session() as session:
            records = list(session.run(cq.cypher, **cq.params))
            project_names = [dict(r["p"]).get("project_name", "") for r in records]
            print(f"  Results: {len(records)} projects -> {project_names}")
    except Exception as e:
        print(f"  FAIL: Cypher execution failed: {e}")
        print(f"  Cypher:\n{cq.cypher}")
        return False, str(e)
    
    # Check if any expected project is in results
    found = []
    for expected in expected_names:
        for actual in project_names:
            if expected.lower() in actual.lower():
                found.append(actual)
                break
    
    if found:
        print(f"  PASS: Found expected project(s): {found}")
        return True, found
    else:
        print(f"  FAIL: None of {expected_names} found in {project_names}")
        print(f"  Cypher:\n{cq.cypher}")
        return False, project_names


if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH RETRIEVAL FIX VERIFICATION")
    print("=" * 70)
    
    results = []
    for question, expected, desc in TEST_CASES:
        passed, details = run_test(question, expected, desc)
        results.append((desc, passed, details))
    
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    for desc, passed, _ in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
    
    print(f"\n{passed_count}/{total} tests passed")
    
    driver.close()
