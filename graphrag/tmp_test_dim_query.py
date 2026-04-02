"""
End-to-end test: verify the LLM prompt generates correct Cypher for dimension queries.
Run from: d:\Rerachatbot\graphrag\ using the graphrag venv.
"""
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, 'D:\\Rerachatbot\\graphrag')
sys.path.insert(0, 'D:\\Rerachatbot')

# Load the project root .env
load_dotenv(os.path.join('D:\\Rerachatbot', '.env'))
load_dotenv(os.path.join('D:\\Rerachatbot', 'graphrag', '.env'))

from graphrag_cypher import generate_cypher

test_cases = [
    # (query, expect_d1_in_params, should_NOT_have_area_min)
    ("find projects with bedroom 10 by 10",          True,  True),
    ("show me projects where bedroom is 10 x 12",    True,  True),
    ("projects with toilet 3 by 6",                  True,  True),
    ("bedroom 10x10 toilet 3x6",                     True,  True),
    ("find bedroom around 120 sqft",                 False, False),  # sqft query — should still use area range
    ("bedroom exactly 100 sqft",                     False, True),   # exact sqft
]

print("=" * 70)
for query, expect_d1, no_area_min in test_cases:
    print(f"\nQuery: {query!r}")
    try:
        result = generate_cypher(query)
        params = result.params
        cypher = result.cypher

        has_d1       = 'd1' in params or 'd1_0' in params or any('d1' in k for k in params)
        has_area_min = 'area_min' in params
        has_lft      = 'length_ft' in cypher

        print(f"  Params:    {params}")
        print(f"  has d1:    {has_d1}   (expected: {expect_d1})")
        print(f"  has area_min: {has_area_min}   (should NOT: {no_area_min})")
        print(f"  length_ft in cypher: {has_lft}")

        ok = True
        if expect_d1 and not has_d1:
            print("  !! FAIL: expected d1 param but not found")
            ok = False
        if no_area_min and has_area_min:
            print("  !! FAIL: should NOT have area_min but it does")
            ok = False
        if ok:
            print("  => PASS")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("-" * 70)
