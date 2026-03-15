"""
test_e2e_queries.py — End-to-end test for LOOKUP & AGGREGATE query types.
Tests the new query types actually execute against Neo4j and return results.

Run: cd d:\Rerachatbot\graphrag && .\\venv\\Scripts\\python.exe test_e2e_queries.py
"""

from graphrag_cypher import generate_cypher
from graphrag_retriever import GraphRetriever

retriever = GraphRetriever()

test_cases = [
    # (description, query, expected_type)
    ("LOOKUP: Details of OUM Orbit Ahmedabad",
     "find me the details of OUM Orbit Ahmedabad",
     "LOOKUP"),

    ("LOOKUP: Hall size in OUM Orbit",
     "what is the hall size in OUM Orbit Ahmedabad",
     "LOOKUP"),

    ("AGGREGATE: Projects with hall > 100 sqft",
     "find projects with hall size greater than 100 sqft",
     "AGGREGATE"),

    ("AGGREGATE: Projects with hall < 100 sqft",
     "find projects with hall size less than 100 sqft",
     "AGGREGATE"),

    ("AGGREGATE: Projects with a study room (existence)",
     "show me projects that have a study room",
     "AGGREGATE"),

    ("AGGREGATE: Projects with pooja room",
     "find projects with a pooja room",
     "AGGREGATE"),

    ("SPECIFIC: 2 BHK in Ahmedabad with pool",
     "2 BHK apartment in Ahmedabad with swimming pool",
     "SPECIFIC"),
]

sep = "=" * 65

for desc, query, expected_type in test_cases:
    print(f"\n{sep}")
    print(f"TEST: {desc}")
    print(f"Query: {query!r}")

    cq = generate_cypher(query)
    print(f"  Type:         {cq.query_type}  {'✓' if cq.query_type == expected_type else '✗ WRONG (expected ' + expected_type + ')'}")
    print(f"  Params:       {list(cq.params.keys())}")
    print(f"  AnsColumns:   {cq.answer_columns}")

    # Execute against Neo4j
    results, answer_data = retriever._execute_cypher(cq)
    print(f"  Projects:     {len(results)}")

    if answer_data:
        print(f"  Answer items: {len(answer_data)}")
        # Show first item preview
        first = answer_data[0] if answer_data else {}
        ut = first.get("unit_type") or first.get("bhk", "")
        rooms = first.get("rooms") or first.get("matching_rooms") or []
        print(f"  First unit:   {ut!r}")
        if rooms:
            r = rooms[0]
            rname = r.get("room_name") or r.get("name", "?")
            area  = r.get("area_sqft")
            dims  = f"{r.get('length')} x {r.get('width')}" if r.get("length") else "—"
            print(f"  First room:   {rname} | {dims} | {area} sqft")
    else:
        if cq.query_type in ("LOOKUP", "AGGREGATE"):
            print(f"  Answer items: 0  (WARN: expected some answer_data)")

print(f"\n{sep}")
print("All tests complete.")
retriever.close()
