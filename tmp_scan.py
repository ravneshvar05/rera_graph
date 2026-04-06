import json, glob

files = sorted(glob.glob(r'd:\Rerachatbot\output\*.json'))
for f in files:
    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)
    bhks = sorted(set(u.get('bhk') for u in d.get('units',[]) if u.get('bhk')))
    ptypes = sorted(set(u.get('property_type') for u in d.get('units',[]) if u.get('property_type')))
    amenities = d.get('amenities', []) or []
    loc = d.get('location', {}) or {}
    sl = d.get('society_layout', {}) or {}
    room_types = set()
    for u in d.get('units',[]):
        for r in u.get('rooms',[]):
            room_types.add(r.get('room_type'))
    print(f"{d.get('project_name')} | city={loc.get('city')} | hood={loc.get('neighbourhood')} | BHK:{bhks} | type:{ptypes} | pool={sl.get('has_swimming_pool')} | club={sl.get('has_clubhouse')} | park={sl.get('has_park_or_garden')} | status={d.get('project_status')} | amenities={len(amenities)} | rooms_types={sorted(room_types)}")
