import json
import glob
import os

folder = 'd:/Rerachatbot/output'
files = glob.glob(os.path.join(folder, '*.json'))

with open('extracted_data.txt', 'w', encoding='utf-8') as out:
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            project_name = data.get('project_name', 'Unknown')
            location = data.get('location', {})
            city = location.get('city', 'Unknown')
            hood = location.get('neighbourhood', 'Unknown')
            address = location.get('address', 'Unknown')
            landmarks = location.get('nearby_landmarks', [])
            amenities = data.get('amenities', [])
            units = data.get('units', [])
            unit_details = []
            for u in units:
                unit_details.append(f"{u.get('bhk')} BHK {u.get('property_type', '')} ({u.get('carpet_area_sqft', 'N/A')} sqft)")
            
            out.write(f"--- {os.path.basename(f)} ---\n")
            out.write(f"Project: {project_name}\n")
            out.write(f"Location: {hood}, {city}\n")
            out.write(f"Address: {address}\n")
            out.write(f"Landmarks: {', '.join(landmarks) if landmarks else 'None'}\n")
            out.write(f"Amenities: {', '.join(amenities) if amenities else 'None'}\n")
            out.write(f"Units: {', '.join(set(unit_details))}\n\n")
