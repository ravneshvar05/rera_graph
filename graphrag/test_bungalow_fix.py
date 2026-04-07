"""
Quick test to verify bungalow/row-house property_type fix.
Run: python test_bungalow_fix.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from graphrag_cypher import _extract_fallback_intent, _post_process_cypher

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

errors = 0

def check(label, actual, expected):
    global errors
    if actual == expected:
        print(f"{PASS} {label}: {actual}")
    else:
        print(f"{FAIL} {label}: got {actual!r}, expected {expected!r}")
        errors += 1

# ── Test 1: Fallback intent – bungalow → list ──────────────────────────────
i = _extract_fallback_intent("find me bungalows in ahmedabad")
check("fallback bungalow", i.property_type, ["VILLA", "BUNGALOW", "ROW_HOUSE"])
check("fallback bungalow city", i.city, "Ahmedabad")

# ── Test 2: Fallback intent – banglow (typo) → list ──────────────────────
i2 = _extract_fallback_intent("banglow in surat")
check("fallback banglow", i2.property_type, ["VILLA", "BUNGALOW", "ROW_HOUSE"])

# ── Test 3: Fallback intent – row house → list ────────────────────────────
i3 = _extract_fallback_intent("row house in ahmedabad")
check("fallback row house", i3.property_type, ["VILLA", "BUNGALOW", "ROW_HOUSE"])

# ── Test 4: Fallback intent – rowhouse (no space) → list ─────────────────
i4 = _extract_fallback_intent("show me rowhouse projects")
check("fallback rowhouse", i4.property_type, ["VILLA", "BUNGALOW", "ROW_HOUSE"])

# ── Test 5: Fallback intent – villa stays single ─────────────────────────
i5 = _extract_fallback_intent("villa in bopal")
check("fallback villa single", i5.property_type, "VILLA")

# ── Test 6: Fallback intent – apartment stays single ─────────────────────
i6 = _extract_fallback_intent("2 BHK flat in nikol")
check("fallback flat→APARTMENT", i6.property_type, "APARTMENT")

# ── Test 7: Post-process safety net – LLM emits single BUNGALOW ──────────
cypher = "WHERE ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))"
params = {"property_type": "BUNGALOW", "city": "Ahmedabad", "limit": 50}
out = _post_process_cypher(cypher, params, "bungalows in ahmedabad")
check("safety-net param key gone",    "property_type" not in params, True)
check("safety-net property_types set", params.get("property_types"), ["VILLA", "BUNGALOW", "ROW_HOUSE"])
check("safety-net cypher rewritten",  "$property_types" in out, True)
check("safety-net no old param in cypher", "$property_type)" not in out, True)

# ── Test 8: Post-process safety net – LLM emits single ROW_HOUSE ─────────
cypher2 = "WHERE ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))"
params2 = {"property_type": "ROW_HOUSE", "limit": 50}
out2 = _post_process_cypher(cypher2, params2, "row house in ahmedabad")
check("safety-net ROW_HOUSE upgraded", params2.get("property_types"), ["VILLA", "BUNGALOW", "ROW_HOUSE"])

# ── Test 9: Post-process safety net – VILLA must NOT be touched ───────────
cypher3 = "WHERE ANY(u IN units WHERE toLower(u.property_type) = toLower($property_type))"
params3 = {"property_type": "VILLA", "limit": 50}
_post_process_cypher(cypher3, params3, "villa in ahmedabad")
check("safety-net VILLA not touched", params3.get("property_type"), "VILLA")
check("safety-net VILLA no property_types", "property_types" not in params3, True)

print()
if errors == 0:
    print(f"{PASS} All {9} tests passed!")
else:
    print(f"{FAIL} {errors} test(s) FAILED")
    sys.exit(1)
