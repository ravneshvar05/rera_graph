import json
from gemini import remove_empty_rooms

def test():
    with open("output/Brochure 17.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned = remove_empty_rooms(data)
    
    for unit in cleaned.get("units", []):
        ut = unit.get("unit_type")
        rooms = unit.get("rooms", [])
        print(f"Unit: {ut} has {len(rooms)} rooms.")
        for r in rooms:
            print(f"  - {r.get('name')}")

if __name__ == "__main__":
    test()
