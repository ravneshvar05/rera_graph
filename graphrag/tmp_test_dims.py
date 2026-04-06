import sys
from graphrag_cypher import _extract_fallback_intent

tests = [
    ('bedroom around 10 by 12', 'around'),
    ('flat with bedroom at least 10 by 12', 'gte'),
    ('show flats with bedroom less than 10 by 12', 'lte'),
    ('bedroom 10 by 12 flat', None),         # exact - no qualifier
    ('2 BHK with kitchen 8x10', None),       # exact
]
all_ok = True
for q, expected_qual in tests:
    intent = _extract_fallback_intent(q)
    dims = intent.room_dimensions
    if dims:
        got = dims[0].get('dim_qualifier')
        status = 'OK' if got == expected_qual else 'FAIL'
        if status == 'FAIL':
            all_ok = False
        print(f'{status}: {q!r} -> dim_qualifier={got!r} (expected {expected_qual!r})')
    else:
        print(f'NODIM: {q!r} -> no room_dimensions extracted')
        all_ok = False
print()
print('All tests passed!' if all_ok else 'SOME TESTS FAILED')
