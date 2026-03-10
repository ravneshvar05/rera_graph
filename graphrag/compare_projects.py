import os
import json

output_dir = r"d:\Rerachatbot\output"
user_projects = [
    "SATYAMEV ELYSIUM",
    "Sunflower Enclave",
    "Svasaar Pravesh",
    "SURYA KUTIR",
    "Satyash Residency",
    "RATNAAKAR 6 Beaumonde",
    "Swastik Harmony",
    "AMBER",
    "Elysium at Shantigram",
    "DWARKESH GREENS",
    "સીતારામ સીટી",
    "Neelkanth Status",
    "NIVAAN GREENS",
    "OUM ORBIT",
    "Sky Avenue",
    "KP Villas",
    "Gandhi Galaxy",
    "Sudama Homes",
    "Palak Elina"
]

# Normalize user project names for comparison
user_projects_norm = [p.strip().lower() for p in user_projects]

found_projects = []
for filename in os.listdir(output_dir):
    if filename.endswith(".json"):
        with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                project_name = data.get("project_name", "Unknown")
                found_projects.append({
                    "name": project_name,
                    "file": filename
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

print(f"Total projects found in files: {len(found_projects)}")
print(f"Total projects in user list: {len(user_projects)}")

print("\nProjects in files NOT in user list:")
for p in found_projects:
    name_norm = p["name"].strip().lower()
    if name_norm not in user_projects_norm:
        # Extra check for substrings or partial matches if needed
        # But let's start with exact match (case insensitive)
        print(f" - {p['name']} (from {p['file']})")

print("\nProjects in user list NOT in files:")
for u_name, u_norm in zip(user_projects, user_projects_norm):
    matches = [f for f in found_projects if f["name"].strip().lower() == u_norm]
    if not matches:
        print(f" - {u_name}")
