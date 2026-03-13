import re

def parse_ft(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    
    # Updated regex
    m = re.match(r"^(\d+)['\u2019](?:[\s\-]?(\d+)[\"\u201d]?)?", raw)
    if m:
        return int(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0)
    
    # 10-6 (feet-inches with dash)
    m = re.match(r"^(\d+)-(\d+)$", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12
    
    # Plain decimal — metres if < 15, else feet
    m = re.match(r"^(\d+\.?\d*)$", raw)
    if m:
        val = float(m.group(1))
        return val * 3.281 if val < 15 else val
    return None

test_cases = [
    "10'-6\"",
    "10'6\"",
    "10'6",
    "10'",
    "17'",
    "17",
    "4.5",
    "10-6"
]

for tc in test_cases:
    print(f"{tc}: {parse_ft(tc)}")
