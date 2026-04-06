"""
Patch script: adds dim_qualifier support for room dimension queries.
Run from d:\\Rerachatbot folder.
"""
import re, sys, codecs

def read(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        return f.read()

def write(path, content):
    with codecs.open(path, 'w', 'utf-8') as f:
        f.write(content)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  graphrag_intent.py  — update room_dimensions field description
# ─────────────────────────────────────────────────────────────────────────────
INTENT_PATH = r'd:\Rerachatbot\graphrag\graphrag_intent.py'
intent = read(INTENT_PATH)

OLD_RDIM = (
    '    room_dimensions: Optional[List[dict]] = Field(\n'
    '        default_factory=list,\n'
    '        description=(\n'
    '            "List of room dimension constraints the user specified. "\n'
    '            "Each entry: {\\"room\\": \\"Bedroom\\", \\"d1\\": 10.6, \\"d2\\": 12.4}. "\n'
    '            "d1/d2 are the two dimensions in feet (feet.inches notation: 12\'6\\" → 12.6). "\n'
    '            "Only populate when user gives explicit NxM / N by M / N x M dimensions for a named room."\n'
    '        )\n'
    '    )'
)

NEW_RDIM = (
    '    room_dimensions: Optional[List[dict]] = Field(\n'
    '        default_factory=list,\n'
    '        description=(\n'
    '            "List of room dimension constraints the user specified. "\n'
    '            "Each entry: {\\"room\\": \\"Bedroom\\", \\"d1\\": 10.6, \\"d2\\": 12.4, \\"dim_qualifier\\": null}. "\n'
    '            "d1/d2 are the two dimensions in feet (feet.inches notation: 12\'6\\" \\u2192 12.6). "\n'
    '            "dim_qualifier: null/\\"exact\\" = exact match (default, no tolerance); "\n'
    '            "\\"around\\" = ±15%% on each dimension independently (NOT area multiplication); "\n'
    '            "\\"gte\\" = both dimensions >= given values (at least / bigger than / minimum); "\n'
    '            "\\"lte\\" = both dimensions <= given values (less than / smaller than / under). "\n'
    '            "Only populate when user gives explicit NxM / N by M / N x M dimensions for a named room."\n'
    '        )\n'
    '    )'
)

if OLD_RDIM in intent:
    intent = intent.replace(OLD_RDIM, NEW_RDIM, 1)
    write(INTENT_PATH, intent)
    print("✅ graphrag_intent.py patched")
else:
    # Try a looser match
    old_snippet = '"Each entry: {\\"room\\": \\"Bedroom\\", \\"d1\\": 10.6, \\"d2\\": 12.4}.'
    if old_snippet in intent:
        new_snippet = (
            '"Each entry: {\\"room\\": \\"Bedroom\\", \\"d1\\": 10.6, \\"d2\\": 12.4, \\"dim_qualifier\\": null}. "\n'
            '            "dim_qualifier: null/\\"exact\\" = exact match (default, no tolerance); "\n'
            '            "\\"around\\" = \\u00b115%% on each dimension independently (NOT area multiplication); "\n'
            '            "\\"gte\\" = both dimensions >= given values (at least / bigger than / minimum); "\n'
            '            "\\"lte\\" = both dimensions <= given values (less than / smaller than / under). "'
        )
        intent = intent.replace(old_snippet, new_snippet, 1)
        # Also fix the line after — remove the old single-line description
        old_after = (
            '"d1/d2 are the two dimensions in feet (feet.inches notation: 12\'6\\" → 12.6). "\n'
            '            "Only populate when user gives explicit NxM / N by M / N x M dimensions for a named room."'
        )
        new_after = (
            '"d1/d2 are the two dimensions in feet (feet.inches notation: 12\'6\\" \\u2192 12.6). "\n'
            '            "Only populate when user gives explicit NxM / N by M / N x M dimensions for a named room."'
        )
        intent = intent.replace(old_after, new_after, 1)
        write(INTENT_PATH, intent)
        print("✅ graphrag_intent.py patched (loose match)")
    else:
        print("⚠️  graphrag_intent.py: could not find target; checking content...")
        idx = intent.find('room_dimensions: Optional')
        print(repr(intent[idx:idx+500]))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  graphrag_cypher.py  — (A) intent schema + (B) dim prompt + (C) fallback extractor
# ─────────────────────────────────────────────────────────────────────────────
CYPHER_PATH = r'd:\Rerachatbot\graphrag\graphrag_cypher.py'
cypher = read(CYPHER_PATH)

# ── 2A: intent schema in LLM prompt (room_dimensions entry) ──────────────────
OLD_SCHEMA_DIM = (
    '    "room_dimensions": [\r\n'
    '      {{"room": "<canonical room name (Bedroom/Kitchen/Hall/etc.)>", "d1": <float feet>, "d2": <float feet>}},\r\n'
    '      ...\r\n'
    '    ]\r\n'
)
NEW_SCHEMA_DIM = (
    '    "room_dimensions": [\r\n'
    '      {{"room": "<canonical room name (Bedroom/Kitchen/Hall/etc.)>", "d1": <float feet>, "d2": <float feet>,\r\n'
    '        "dim_qualifier": null | "exact" | "around" | "gte" | "lte"}},\r\n'
    '      ...\r\n'
    '    ]\r\n'
)

if OLD_SCHEMA_DIM in cypher:
    cypher = cypher.replace(OLD_SCHEMA_DIM, NEW_SCHEMA_DIM, 1)
    print("✅ cypher prompt intent schema patched")
else:
    # Try without \r
    OLD_SCHEMA_DIM2 = (
        '    "room_dimensions": [\n'
        '      {{"room": "<canonical room name (Bedroom/Kitchen/Hall/etc.)>", "d1": <float feet>, "d2": <float feet>}},\n'
        '      ...\n'
        '    ]\n'
    )
    NEW_SCHEMA_DIM2 = (
        '    "room_dimensions": [\n'
        '      {{"room": "<canonical room name (Bedroom/Kitchen/Hall/etc.)>", "d1": <float feet>, "d2": <float feet>,\n'
        '        "dim_qualifier": null | "exact" | "around" | "gte" | "lte"}},\n'
        '      ...\n'
        '    ]\n'
    )
    if OLD_SCHEMA_DIM2 in cypher:
        cypher = cypher.replace(OLD_SCHEMA_DIM2, NEW_SCHEMA_DIM2, 1)
        print("✅ cypher prompt intent schema patched (LF)")
    else:
        print("⚠️  intent schema dim section not found exactly")
        idx = cypher.find('"room_dimensions"')
        print(repr(cypher[idx:idx+300]))

# ── 2B: dimension section in LLM prompt — add qualifier instructions after step 4 ──
# The EXACT match step 4 ends with: "}}" and then "  Step 5 — query_type=AGGREGATE"
OLD_DIM_STEPS = (
    '  Step 4 — Generate orientation-agnostic EXACT filter on length_ft / width_ft:\r\n'
    '    WHERE EXISTS {{\r\n'
    '      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room)\r\n'
    '      WHERE r2.name IN $room_names\r\n'
    '      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL\r\n'
    '      AND (\r\n'
    '        (toFloat(r2.length_ft) = toFloat($d1) AND toFloat(r2.width_ft) = toFloat($d2))\r\n'
    '        OR (toFloat(r2.length_ft) = toFloat($d2) AND toFloat(r2.width_ft) = toFloat($d1))\r\n'
    '      )\r\n'
    '    }}\r\n'
    '  Step 5 — query_type=AGGREGATE. Include answer_data with matching_rooms.'
)

LF = '\r\n'
NEW_DIM_STEPS = (
    f'  Step 3 — Detect qualifier word (if any) in the user query:{LF}'
    f'    "around/approximately/roughly/about/~/close to/near about" → dim_qualifier="around"{LF}'
    f'    "at least/minimum/bigger than/larger than/more than/atleast/>=" → dim_qualifier="gte"{LF}'
    f'    "less than/smaller than/under/<=" → dim_qualifier="lte"{LF}'
    f'    No qualifier word present → dim_qualifier=null  (exact match, DEFAULT){LF}'
    f'  Step 4 — Generate the WHERE clause based on dim_qualifier:{LF}'
    f'{LF}'
    f'  4a. EXACT (dim_qualifier=null, default — NO qualifier word in query):{LF}'
    f'    WHERE EXISTS {{{{{LF}'
    f'      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room){LF}'
    f'      WHERE r2.name IN $room_names{LF}'
    f'      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL{LF}'
    f'      AND ({LF}'
    f'        (toFloat(r2.length_ft) = toFloat($d1) AND toFloat(r2.width_ft) = toFloat($d2)){LF}'
    f'        OR (toFloat(r2.length_ft) = toFloat($d2) AND toFloat(r2.width_ft) = toFloat($d1)){LF}'
    f'      ){LF}'
    f'    }}}}{LF}'
    f'    Params: {{d1, d2, room_names}}{LF}'
    f'{LF}'
    f'  4b. AROUND (dim_qualifier="around" — ±15%% on each dim independently, NOT area):{LF}'
    f'    Compute: d1_lo=d1*0.85, d1_hi=d1*1.15, d2_lo=d2*0.85, d2_hi=d2*1.15{LF}'
    f'    WHERE EXISTS {{{{{LF}'
    f'      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room){LF}'
    f'      WHERE r2.name IN $room_names{LF}'
    f'      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL{LF}'
    f'      AND ({LF}'
    f'        (toFloat(r2.length_ft) >= toFloat($d1_lo) AND toFloat(r2.length_ft) <= toFloat($d1_hi){LF}'
    f'         AND toFloat(r2.width_ft) >= toFloat($d2_lo) AND toFloat(r2.width_ft) <= toFloat($d2_hi)){LF}'
    f'        OR{LF}'
    f'        (toFloat(r2.length_ft) >= toFloat($d2_lo) AND toFloat(r2.length_ft) <= toFloat($d2_hi){LF}'
    f'         AND toFloat(r2.width_ft) >= toFloat($d1_lo) AND toFloat(r2.width_ft) <= toFloat($d1_hi)){LF}'
    f'      ){LF}'
    f'    }}}}{LF}'
    f'    Params: {{d1_lo, d1_hi, d2_lo, d2_hi, room_names}} (NO d1/d2 raw params for around){LF}'
    f'    CRITICAL: ±15%% applies to each dimension INDEPENDENTLY. NEVER multiply d1*d2 to get area.{LF}'
    f'{LF}'
    f'  4c. GTE (dim_qualifier="gte" — both dims >= given values, orientation-agnostic):{LF}'
    f'    WHERE EXISTS {{{{{LF}'
    f'      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room){LF}'
    f'      WHERE r2.name IN $room_names{LF}'
    f'      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL{LF}'
    f'      AND ({LF}'
    f'        (toFloat(r2.length_ft) >= toFloat($d1) AND toFloat(r2.width_ft) >= toFloat($d2)){LF}'
    f'        OR (toFloat(r2.length_ft) >= toFloat($d2) AND toFloat(r2.width_ft) >= toFloat($d1)){LF}'
    f'      ){LF}'
    f'    }}}}{LF}'
    f'    Params: {{d1, d2, room_names}}{LF}'
    f'{LF}'
    f'  4d. LTE (dim_qualifier="lte" — both dims <= given values, orientation-agnostic):{LF}'
    f'    WHERE EXISTS {{{{{LF}'
    f'      MATCH (p)-[:HAS_UNIT]->(u2:Unit)-[:HAS_ROOM]->(r2:Room){LF}'
    f'      WHERE r2.name IN $room_names{LF}'
    f'      AND r2.length_ft IS NOT NULL AND r2.width_ft IS NOT NULL{LF}'
    f'      AND ({LF}'
    f'        (toFloat(r2.length_ft) <= toFloat($d1) AND toFloat(r2.width_ft) <= toFloat($d2)){LF}'
    f'        OR (toFloat(r2.length_ft) <= toFloat($d2) AND toFloat(r2.width_ft) <= toFloat($d1)){LF}'
    f'      ){LF}'
    f'    }}}}{LF}'
    f'    Params: {{d1, d2, room_names}}{LF}'
    f'{LF}'
    f'  Step 5 — query_type=AGGREGATE. Include answer_data with matching_rooms.'
)

# Try with \r\n first
if OLD_DIM_STEPS in cypher:
    # Remove the now-redundant old Step 3 (set params d1 and d2) and old Step 4 header
    OLD_FULL = (
        '  Step 1 — Identify the room type and the two numbers (in feet).\r\n'
        '  Step 2 — DO NOT multiply. DO NOT convert to area. DO NOT use area_sqft.\r\n'
        '  Step 3 — Set params d1 and d2 as float values: "10 by 12" → d1=10.0, d2=12.0\r\n'
        + OLD_DIM_STEPS
    )
    NEW_FULL = (
        '  Step 1 — Identify the room type and the two numbers (in feet).\r\n'
        '  Step 2 — DO NOT multiply. DO NOT convert to area. DO NOT use area_sqft.\r\n'
        '  Step 3 — Set params d1 and d2 as float values: "10 by 12" → d1=10.0, d2=12.0\r\n'
        + NEW_DIM_STEPS
    )
    if OLD_FULL in cypher:
        # Replace old steps 3+4 with new steps 3+4 (keeping steps 1+2)
        OLD_S3_S4 = (
            '  Step 3 — Set params d1 and d2 as float values: "10 by 12" → d1=10.0, d2=12.0\r\n'
            + OLD_DIM_STEPS
        )
        NEW_S3_S4 = NEW_DIM_STEPS
        cypher = cypher.replace(OLD_S3_S4, NEW_S3_S4, 1)
        print("✅ cypher dimension steps patched (CRLF)")
    else:
        cypher = cypher.replace(OLD_DIM_STEPS, NEW_DIM_STEPS, 1)
        print("✅ cypher dimension step 4 patched (CRLF)")
else:
    # Try LF only
    OLD_DIM_STEPS_LF = OLD_DIM_STEPS.replace('\r\n', '\n')
    NEW_DIM_STEPS_LF = NEW_DIM_STEPS.replace('\r\n', '\n')
    if OLD_DIM_STEPS_LF in cypher:
        OLD_S3_S4_LF = (
            '  Step 3 — Set params d1 and d2 as float values: "10 by 12" → d1=10.0, d2=12.0\n'
            + OLD_DIM_STEPS_LF
        )
        cypher = cypher.replace(OLD_S3_S4_LF, NEW_DIM_STEPS_LF, 1)
        print("✅ cypher dimension steps patched (LF)")
    else:
        print("⚠️  dimension steps not found; checking...")
        idx = cypher.find('Step 4 — Generate orientation-agnostic EXACT')
        print(repr(cypher[max(0,idx-200):idx+400]))

# ── 2B2: Update the intent example for room_dimensions to add dim_qualifier ──
OLD_INTENT_EX = (
    '           intent.room_dimensions = [{{\"room\": \"Bedroom\", \"d1\": 10.6, \"d2\": 12.4}}]'
)
NEW_INTENT_EX = (
    '           intent.room_dimensions = [{{\"room\": \"Bedroom\", \"d1\": 10.6, \"d2\": 12.4, \"dim_qualifier\": null}}]\r\n'
    '           # "around bedroom 10 by 12": dim_qualifier="around" → ±15%% on each dim\r\n'
    '           # "bedroom at least 10 by 12": dim_qualifier="gte"\r\n'
    '           # "bedroom less than 10 by 12": dim_qualifier="lte"'
)
if OLD_INTENT_EX in cypher:
    cypher = cypher.replace(OLD_INTENT_EX, NEW_INTENT_EX, 1)
    print("✅ cypher intent example patched")
else:
    OLD_INTENT_EX_LF = OLD_INTENT_EX.replace('\r\n', '\n')
    NEW_INTENT_EX_LF = NEW_INTENT_EX.replace('\r\n', '\n')
    if OLD_INTENT_EX_LF in cypher:
        cypher = cypher.replace(OLD_INTENT_EX_LF, NEW_INTENT_EX_LF, 1)
        print("✅ cypher intent example patched (LF)")
    else:
        print("⚠️  intent example not found")
        idx = cypher.find('intent.room_dimensions = ')
        print(repr(cypher[max(0,idx-20):idx+150]))

# ── 2C: Fallback extractor — detect qualifier near dimension, store in dict ──
OLD_APPEND = (
    '                    room_dimensions.append({"room": canonical, "d1": d1, "d2": d2})'
)
NEW_APPEND = r'''                    # Detect qualifier word near this dimension match
                    _prefix = lower_query[:match.end()]
                    _dim_qualifier = None
                    _AROUND = ('around', 'approximately', 'roughly', 'about', 'approx', 'close to', 'near about', '~')
                    _GTE    = ('at least', 'atleast', 'minimum', 'bigger than', 'larger than', 'more than', 'at-least')
                    _LTE    = ('less than', 'smaller than', 'under', 'no more than')
                    if any(w in _prefix for w in _AROUND):
                        _dim_qualifier = 'around'
                    elif any(w in _prefix for w in _GTE):
                        _dim_qualifier = 'gte'
                    elif any(w in _prefix for w in _LTE):
                        _dim_qualifier = 'lte'
                    room_dimensions.append({"room": canonical, "d1": d1, "d2": d2, "dim_qualifier": _dim_qualifier})'''

if OLD_APPEND in cypher:
    cypher = cypher.replace(OLD_APPEND, NEW_APPEND, 1)
    print("✅ fallback extractor dim_qualifier detection patched")
else:
    OLD_APPEND_LF = OLD_APPEND.replace('\r\n', '\n')
    NEW_APPEND_LF = NEW_APPEND.replace('\r\n', '\n')
    if OLD_APPEND_LF in cypher:
        cypher = cypher.replace(OLD_APPEND_LF, NEW_APPEND_LF, 1)
        print("✅ fallback extractor patched (LF)")
    else:
        print("⚠️  fallback extractor append not found")
        idx = cypher.find('room_dimensions.append(')
        print(repr(cypher[max(0,idx-30):idx+150]))

write(CYPHER_PATH, cypher)
print("✅ graphrag_cypher.py written")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  graphrag_retriever.py  — qualifier-aware Cypher for room dims
# ─────────────────────────────────────────────────────────────────────────────
RET_PATH = r'd:\Rerachatbot\graphrag\graphrag_retriever.py'
ret = read(RET_PATH)

OLD_DIMBLOCK = (
    '        # ── Room dimensions — added as EXISTS subqueries (one per room) ────────\r\n'
    '        room_dims = getattr(intent, "room_dimensions", None) or []\r\n'
    '        for i, rd in enumerate(room_dims):\r\n'
    '            raw_room = rd.get("room", "")\r\n'
    '            d1 = rd.get("d1")\r\n'
    '            d2 = rd.get("d2")\r\n'
    '            if not raw_room or d1 is None or d2 is None:\r\n'
    '                continue\r\n'
    '            # Expand room name to canonical synonyms\r\n'
    '            from graphrag_cypher import _expand_room_name\r\n'
    '            canonical_names = _expand_room_name(raw_room)\r\n'
    '            rnames_key = f"dim_room_names_{i}"\r\n'
    '            d1_key = f"dim_d1_{i}"\r\n'
    '            d2_key = f"dim_d2_{i}"\r\n'
    '            params[rnames_key] = canonical_names\r\n'
    '            params[d1_key] = float(d1)\r\n'
    '            params[d2_key] = float(d2)\r\n'
    '            # We can\'t use $list in a WHERE with IN easily via the fallback Cypher\r\n'
    '            # build it as: r2.name IN [\'Bedroom\', \'Master Bedroom\']\r\n'
    '            names_literal = json.dumps(canonical_names)\r\n'
    '            where_clauses.append(\r\n'
    '                f"EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u_dim_{i}:Unit)-[:HAS_ROOM]->(r_dim_{i}:Room) "\r\n'
    '                f"WHERE r_dim_{i}.name IN {names_literal} "\r\n'
    '                f"AND r_dim_{i}.length_ft IS NOT NULL AND r_dim_{i}.width_ft IS NOT NULL "\r\n'
    '                f"AND ((toFloat(r_dim_{i}.length_ft) = toFloat(${d1_key}) AND toFloat(r_dim_{i}.width_ft) = toFloat(${d2_key})) "\r\n'
    '                f"OR (toFloat(r_dim_{i}.length_ft) = toFloat(${d2_key}) AND toFloat(r_dim_{i}.width_ft) = toFloat(${d1_key}))) }}"\r\n'
    '            )\r\n'
    '            logger.debug(\r\n'
    '                f"[Fallback] Added room-dimension filter: {canonical_names} {d1}x{d2}"\r\n'
    '            )'
)

NEW_DIMBLOCK = r'''        # ── Room dimensions — EXISTS subqueries, qualifier-aware ──────────────
        # dim_qualifier: None/"exact" → exact match (default)
        #                "around"     → ±15% on each dim individually (NOT area)
        #                "gte"        → both dims >= values (at least / bigger than)
        #                "lte"        → both dims <= values (less than / smaller than)
        _DIM_TOLERANCE = 0.15
        room_dims = getattr(intent, "room_dimensions", None) or []
        for i, rd in enumerate(room_dims):
            raw_room = rd.get("room", "")
            d1 = rd.get("d1")
            d2 = rd.get("d2")
            dim_qualifier = rd.get("dim_qualifier") or None   # None/"exact" → exact
            if not raw_room or d1 is None or d2 is None:
                continue
            from graphrag_cypher import _expand_room_name
            canonical_names = _expand_room_name(raw_room)
            names_literal = json.dumps(canonical_names)
            d1_f, d2_f = float(d1), float(d2)

            if dim_qualifier == "around":
                # ±15% on each dimension independently — NOT area multiplication
                d1_lo = round(d1_f * (1 - _DIM_TOLERANCE), 4)
                d1_hi = round(d1_f * (1 + _DIM_TOLERANCE), 4)
                d2_lo = round(d2_f * (1 - _DIM_TOLERANCE), 4)
                d2_hi = round(d2_f * (1 + _DIM_TOLERANCE), 4)
                k = {
                    "d1_lo": f"dim_d1_lo_{i}", "d1_hi": f"dim_d1_hi_{i}",
                    "d2_lo": f"dim_d2_lo_{i}", "d2_hi": f"dim_d2_hi_{i}",
                }
                params[k["d1_lo"]] = d1_lo; params[k["d1_hi"]] = d1_hi
                params[k["d2_lo"]] = d2_lo; params[k["d2_hi"]] = d2_hi
                where_clauses.append(
                    f"EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u_dim_{i}:Unit)-[:HAS_ROOM]->(r_dim_{i}:Room) "
                    f"WHERE r_dim_{i}.name IN {names_literal} "
                    f"AND r_dim_{i}.length_ft IS NOT NULL AND r_dim_{i}.width_ft IS NOT NULL "
                    f"AND ("
                    f"(toFloat(r_dim_{i}.length_ft) >= toFloat(${k['d1_lo']}) AND toFloat(r_dim_{i}.length_ft) <= toFloat(${k['d1_hi']}) "
                    f"AND toFloat(r_dim_{i}.width_ft) >= toFloat(${k['d2_lo']}) AND toFloat(r_dim_{i}.width_ft) <= toFloat(${k['d2_hi']})) "
                    f"OR "
                    f"(toFloat(r_dim_{i}.length_ft) >= toFloat(${k['d2_lo']}) AND toFloat(r_dim_{i}.length_ft) <= toFloat(${k['d2_hi']}) "
                    f"AND toFloat(r_dim_{i}.width_ft) >= toFloat(${k['d1_lo']}) AND toFloat(r_dim_{i}.width_ft) <= toFloat(${k['d1_hi']}))) }}"
                )
                logger.debug(f"[Fallback] Dim filter AROUND: {canonical_names} {d1}x{d2} (±{int(_DIM_TOLERANCE*100)}%)")

            elif dim_qualifier == "gte":
                # Both dims >= given values, orientation-agnostic
                d1_k, d2_k = f"dim_d1_{i}", f"dim_d2_{i}"
                params[d1_k] = d1_f; params[d2_k] = d2_f
                where_clauses.append(
                    f"EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u_dim_{i}:Unit)-[:HAS_ROOM]->(r_dim_{i}:Room) "
                    f"WHERE r_dim_{i}.name IN {names_literal} "
                    f"AND r_dim_{i}.length_ft IS NOT NULL AND r_dim_{i}.width_ft IS NOT NULL "
                    f"AND ("
                    f"(toFloat(r_dim_{i}.length_ft) >= toFloat(${d1_k}) AND toFloat(r_dim_{i}.width_ft) >= toFloat(${d2_k})) "
                    f"OR (toFloat(r_dim_{i}.length_ft) >= toFloat(${d2_k}) AND toFloat(r_dim_{i}.width_ft) >= toFloat(${d1_k}))) }}"
                )
                logger.debug(f"[Fallback] Dim filter GTE: {canonical_names} >= {d1}x{d2}")

            elif dim_qualifier == "lte":
                # Both dims <= given values, orientation-agnostic
                d1_k, d2_k = f"dim_d1_{i}", f"dim_d2_{i}"
                params[d1_k] = d1_f; params[d2_k] = d2_f
                where_clauses.append(
                    f"EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u_dim_{i}:Unit)-[:HAS_ROOM]->(r_dim_{i}:Room) "
                    f"WHERE r_dim_{i}.name IN {names_literal} "
                    f"AND r_dim_{i}.length_ft IS NOT NULL AND r_dim_{i}.width_ft IS NOT NULL "
                    f"AND ("
                    f"(toFloat(r_dim_{i}.length_ft) <= toFloat(${d1_k}) AND toFloat(r_dim_{i}.width_ft) <= toFloat(${d2_k})) "
                    f"OR (toFloat(r_dim_{i}.length_ft) <= toFloat(${d2_k}) AND toFloat(r_dim_{i}.width_ft) <= toFloat(${d1_k}))) }}"
                )
                logger.debug(f"[Fallback] Dim filter LTE: {canonical_names} <= {d1}x{d2}")

            else:
                # EXACT match (default — no qualifier word, preserves original behavior)
                d1_k, d2_k = f"dim_d1_{i}", f"dim_d2_{i}"
                params[d1_k] = d1_f; params[d2_k] = d2_f
                where_clauses.append(
                    f"EXISTS {{ MATCH (p)-[:HAS_UNIT]->(u_dim_{i}:Unit)-[:HAS_ROOM]->(r_dim_{i}:Room) "
                    f"WHERE r_dim_{i}.name IN {names_literal} "
                    f"AND r_dim_{i}.length_ft IS NOT NULL AND r_dim_{i}.width_ft IS NOT NULL "
                    f"AND ((toFloat(r_dim_{i}.length_ft) = toFloat(${d1_k}) AND toFloat(r_dim_{i}.width_ft) = toFloat(${d2_k})) "
                    f"OR (toFloat(r_dim_{i}.length_ft) = toFloat(${d2_k}) AND toFloat(r_dim_{i}.width_ft) = toFloat(${d1_k}))) }}"
                )
                logger.debug(f"[Fallback] Dim filter EXACT: {canonical_names} {d1}x{d2}")'''

if OLD_DIMBLOCK in ret:
    ret = ret.replace(OLD_DIMBLOCK, NEW_DIMBLOCK, 1)
    print("✅ retriever dim block patched (CRLF)")
else:
    OLD_DIMBLOCK_LF = OLD_DIMBLOCK.replace('\r\n', '\n')
    if OLD_DIMBLOCK_LF in ret:
        ret = ret.replace(OLD_DIMBLOCK_LF, NEW_DIMBLOCK, 1)
        print("✅ retriever dim block patched (LF)")
    else:
        print("⚠️  retriever dim block not found; checking...")
        idx = ret.find('# ── Room dimensions — added as EXISTS')
        print(repr(ret[idx:idx+300]))
        # Show actual content so we can match it
        idx2 = ret.find('room_dims = getattr(intent, "room_dimensions"')
        print("\n--- actual content ---")
        print(repr(ret[max(0,idx2-80):idx2+800]))

write(RET_PATH, ret)
print("✅ graphrag_retriever.py written")

print("\n=== All patches applied ===")
