import os
import json

def main():
    output_dir = "output"
    facts_file = "facts.txt"
    
    with open(facts_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(output_dir):
            if not filename.endswith(".json"):
                continue
            
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "r", encoding="utf-8") as jf:
                try:
                    data = json.load(jf)
                except:
                    print(f"Failed to load {filename}")
                    continue
                
                f.write(f"--- {filepath} ---\n")
                f.write(f"Project: {data.get('project_name')}\n")
                
                loc = data.get("location", {})
                f.write(f"Location: {loc.get('neighbourhood')}, {loc.get('city')}\n")
                f.write(f"Landmarks: {loc.get('nearby_landmarks', [])[:5]}\n")
                
                amenities = data.get("amenities", [])
                f.write(f"Amenities: {amenities[:10]}\n")
                
                units = data.get("units", [])
                unit_types = set([f"{u.get('bhk')} BHK {u.get('property_type')}" for u in units if u.get('bhk') and u.get('property_type')])
                f.write(f"Units: {list(unit_types)}\n")
                
                society = data.get("society_layout", {})
                villas = society.get("total_independent_villas_or_tenements")
                f.write(f"Villas: {villas}\n")
                
                has_club = society.get("has_clubhouse")
                has_pool = society.get("has_swimming_pool")
                has_park = society.get("has_park_or_garden")
                f.write(f"Features: Club={has_club}, Pool={has_pool}, Park={has_park}\n")
                
                f.write("\n")
                
    print(f"Facts written to {facts_file}")

if __name__ == "__main__":
    main()
