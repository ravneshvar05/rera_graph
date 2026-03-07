import json

files = ['output/Brochure 11.json', 'output/Brochure 13.json', 'output/Brochure2.json']

for f in files:
    with open(f, 'r', encoding='utf-8') as f_obj:
        data = json.load(f_obj)
        print(f"--- Project: {data.get('project_name', 'Unknown')} ({f}) ---")
        soc = data.get('society_layout', {})
        print(f"Has Clubhouse: {soc.get('has_clubhouse')}")
        print(f"Amenities list: {data.get('amenities', [])}")
        print(f"Units list count: {len(data.get('units', []))}")
