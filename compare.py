import json
import sys

def analyze_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compare_jsons(old_data, new_data):
    def count_empty(obj):
        empty_count = 0
        total_count = 0
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                e, t = count_empty(v)
                empty_count += e
                total_count += t + 1
        elif isinstance(obj, list):
            for item in obj:
                e, t = count_empty(item)
                empty_count += e
                total_count += t + 1
            if len(obj) == 0:
                empty_count += 1
        else:
            if obj is None or obj == "" or str(obj).strip().lower() in ["na", "n/a", "not available", "unknown", "none"]:
                empty_count += 1
        return empty_count, total_count

    print("=== OVERALL METRICS ===")
    old_empty, old_total = count_empty(old_data)
    new_empty, new_total = count_empty(new_data)
    print(f"Old JSON: {old_empty} empty/NA fields out of {old_total} total fields")
    print(f"New JSON: {new_empty} empty/NA fields out of {new_total} total fields")
    
    # Specific field comparisons
    print("\n=== FIELD COMPARISONS ===")
    
    def compare_dict(d1, d2, path=""):
        if isinstance(d1, dict) and isinstance(d2, dict):
            all_keys = set(d1.keys()).union(set(d2.keys()))
            for k in sorted(all_keys):
                new_path = f"{path}.{k}" if path else k
                v1 = d1.get(k)
                v2 = d2.get(k)
                
                # If both are dicts or lists, recurse (for lists, just check length for now if simple)
                if isinstance(v1, dict) and isinstance(v2, dict):
                    compare_dict(v1, v2, new_path)
                elif isinstance(v1, list) and isinstance(v2, list):
                    l1, l2 = len(v1), len(v2)
                    if l1 != l2:
                        print(f"[CHANGE] {new_path}: Old had {l1} items, New has {l2} items")
                    else:
                        # Compare first item if dict
                        if l1 > 0 and isinstance(v1[0], dict) and isinstance(v2[0], dict):
                            compare_dict(v1[0], v2[0], f"{new_path}[0]")
                else:
                    if str(v1) != str(v2):
                        # only print if one is empty and other is not, or meaningful change
                        is_emp1 = v1 is None or v1 == "" or str(v1).lower() in ["na", "n/a", "none"]
                        is_emp2 = v2 is None or v2 == "" or str(v2).lower() in ["na", "n/a", "none"]
                        
                        if is_emp1 and not is_emp2:
                            print(f"[IMPROVEMENT] {new_path}: (Old: Empty/NA) -> (New: {str(v2)[:50]}...)")
                        elif not is_emp1 and is_emp2:
                            print(f"[DEGRADATION]  {new_path}: (Old: {str(v1)[:50]}...) -> (New: Empty/NA)")
                        else:
                            print(f"[DIFFERENCE]  {new_path}:\n  Old: {str(v1)[:50]}...\n  New: {str(v2)[:50]}...")

    compare_dict(old_data, new_data)

if __name__ == "__main__":
    old_file = r"d:\Rerachatbot\output\old.json"
    new_file = r"d:\Rerachatbot\output\KP_Villa Bochure.json"
    
    try:
        old_data = analyze_json(old_file)
        new_data = analyze_json(new_file)
        compare_jsons(old_data, new_data)
    except Exception as e:
        print(f"Error: {e}")
