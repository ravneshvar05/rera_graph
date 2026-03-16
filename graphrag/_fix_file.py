"""Temporary script to fix graphrag_cypher.py by keeping lines 1-808 and appending correct tail."""
import pathlib

src = pathlib.Path("graphrag_cypher.py")
lines = src.read_text(encoding="utf-8").splitlines(keepends=True)

# Keep only lines 1-808 (index 0-807)
good = lines[:808]

tail = r'''
# ── Main generation function ───────────────────────────────────────────────────

def generate_cypher(user_query: str) -> CypherQuery:
    """
    Generate a precise Cypher query from the user's natural language query.
    """
    try:
        response = _groq.chat.completions.create(
            model=settings.GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        cypher = data.get("cypher", "").strip()
        params = data.get("params", {})
        query_type = data.get("query_type", "SPECIFIC")
        vector_query = data.get("vector_query", user_query)
        answer_columns = data.get("answer_columns", [])

        # Safety: ensure limit param is present
        if "limit" not in params:
            params["limit"] = 50

        # Safety: LOOKUP/AGGREGATE must declare answer_columns
        if query_type in ("LOOKUP", "AGGREGATE") and not answer_columns:
            answer_columns = ["answer_data"]

        if not cypher:
            raise ValueError("LLM returned empty cypher")

        # Post-process: fix common LLM Cypher mistakes
        cypher = _post_process_cypher(cypher, params)

        logger.info(
            f"[Text-to-Cypher] type={query_type} | params={list(params.keys())} | "
            f"answer_cols={answer_columns} | vector_query={vector_query!r}"
        )
        logger.debug(f"[Text-to-Cypher] Cypher:\n{cypher}")

        return CypherQuery(
            cypher=cypher,
            params=params,
            query_type=query_type,
            vector_query=vector_query,
            answer_columns=answer_columns,
        )

    except Exception as e:
        logger.warning(f"[Text-to-Cypher] LLM generation failed ({e}). Using fallback.")
        return _fallback_cypher()


def _post_process_cypher(cypher: str, params: dict) -> str:
    """
    Fix common LLM-generated Cypher issues:
    - Ensure toFloat() wraps area_sqft comparisons
    - Fix toLower/toFloat/toInteger applied to list params
    - Expand room name params to synonym groups  (generalized)
    - Expand amenity tag params to synonym groups (generalized)
    """
    # 1. Numeric type safety
    for sqft_field in ('area_sqft', 'carpet_sqft', 'super_builtup_sqft'):
        cypher = re.sub(
            r'(?<!toFloat\()([a-zA-Z]\w*\.' + sqft_field + r')\s*([><!=]+)\s*(?!toFloat)',
            lambda m: f'toFloat({m.group(1)}) {m.group(2)} toFloat',
            cypher,
        )

    # 2. Scalar-function-on-list protection
    cypher = re.sub(
        r'\bIN\s+toLower\(\$(\w+)\)',
        lambda m: f'IN [x IN ${m.group(1)} | toLower(x)]',
        cypher,
    )

    for pname, pval in params.items():
        if not isinstance(pval, list):
            continue
        for func in ('toLower', 'toFloat', 'toInteger'):
            pat = re.compile(r'\b' + func + r'\(\$' + re.escape(pname) + r'\)')
            if pat.search(cypher):
                cypher = pat.sub(f'[x IN ${pname} | {func}(x)]', cypher)
                logger.debug(f"[Post-process] Fixed {func} on list param ${pname}")

    # 3. Generalized room synonym expansion
    for pname in list(params.keys()):
        pval = params[pname]
        if 'room' not in pname.lower():
            continue

        if isinstance(pval, str):
            expanded = _expand_room_name(pval)
            if len(expanded) > 1:
                new_pname = pname if pname.endswith('s') else pname + 's'
                params[new_pname] = expanded
                cypher = re.sub(
                    r'(\w+\.name)\s*=\s*\$' + re.escape(pname) + r'\b',
                    r'\1 IN $' + new_pname,
                    cypher,
                )
                cypher = re.sub(
                    r'toLower\((\w+\.name)\)\s*=\s*toLower\(\$' + re.escape(pname) + r'\)',
                    r'\1 IN $' + new_pname,
                    cypher,
                )
                if new_pname != pname and pname in params:
                    del params[pname]
                logger.debug(f"[Post-process] Expanded room '{pval}' -> {expanded}")

        elif isinstance(pval, list):
            all_expanded: list[str] = []
            for name in pval:
                for cn in _expand_room_name(name):
                    if cn not in all_expanded:
                        all_expanded.append(cn)
            if all_expanded != pval:
                params[pname] = all_expanded
                logger.debug(f"[Post-process] Expanded room list -> {all_expanded}")

    # 4. Generalized amenity tag synonym expansion
    for pname in list(params.keys()):
        pval = params[pname]
        if not ('tag' in pname.lower() or 'amenity' in pname.lower()):
            continue

        if isinstance(pval, str):
            expanded = _expand_amenity_tags(pval)
            if len(expanded) > 1:
                new_pname = 'amenity_tags' if pname == 'tag' else pname
                params[new_pname] = expanded
                cypher = re.sub(
                    r'\$' + re.escape(pname) + r'\s+IN\s+(\w+\.canonical_tags)',
                    r'ANY(t IN $' + new_pname + r' WHERE t IN \1)',
                    cypher,
                )
                if new_pname != pname and new_pname in params:
                    del params[pname]
                logger.debug(f"[Post-process] Expanded amenity '{pval}' -> {expanded}")
            elif len(expanded) == 1 and expanded[0] != pval:
                params[pname] = expanded[0]
                logger.debug(f"[Post-process] Normalized amenity '{pval}' -> '{expanded[0]}'")

        elif isinstance(pval, list):
            all_tags: list[str] = []
            for t in pval:
                for tag in _expand_amenity_tags(t):
                    if tag not in all_tags:
                        all_tags.append(tag)
            if all_tags != pval:
                params[pname] = all_tags
                logger.debug(f"[Post-process] Expanded amenity tags -> {all_tags}")

    return cypher


# Quick test
if __name__ == "__main__":
    test_queries = [
        "3 BHK flat in Ahmedabad with pooja room",
        "east-facing villa with swimming pool",
        "what is the hall size in OUM Orbit Ahmedabad",
        "kitchen dimensions in Svasaar Pravesh",
        "find me the details of svasaar pravesh",
        "does OUM Orbit have swimming pool?",
        "find projects with hall size greater than 100 sqft",
        "show me projects that have a study room",
        "how many projects are in Ahmedabad?",
        "which project has the largest hall?",
        "tell me about OUM Orbit",
        "2 BHK in Vinzol",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = generate_cypher(q)
        print(f"Type:   {result.query_type}")
        print(f"Params: {result.params}")
        print(f"AnsCol: {result.answer_columns}")
        print(f"Vector: {result.vector_query}")
        print(f"Cypher:\n{result.cypher}")
'''

src.write_text("".join(good) + tail, encoding="utf-8")
print("Done! File rewritten.")

# Verify it compiles
import py_compile
py_compile.compile(str(src), doraise=True)
print("File compiles OK!")
