import os
import random
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv('d:/Rerachatbot/kg/.env')

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Brochures with known complex layouts 
TEST_PROJECTS = ["OumOrbit_Brochure_1", "sitaram_city_apartments"]

def verify_new_fields():
    with driver.session() as session:
        for pid in TEST_PROJECTS:
            print(f"\n======================================")
            print(f" VERIFYING PROJECT: {pid}")
            print(f"======================================")
            
            # 1. Check Project Node New Fields
            cypher_proj = """
            MATCH (p:Project {project_id: $pid})
            RETURN p.total_buildings AS blocks, p.total_villas AS villas, 
                   p.road_widths AS roads, p.building_names AS buildings
            """
            proj = session.run(cypher_proj, pid=pid).single()
            if proj:
                print("\n--- Project Properties ---")
                print(f"Total Blocks: {proj['blocks']}")
                print(f"Total Villas: {proj['villas']}")
                print(f"Building Names (Array): {proj['buildings']}")
                print(f"Road Widths (Array): {proj['roads']}")
            
            # 2. Check Floor Layouts
            cypher_floors = """
            MATCH (p:Project {project_id: $pid})-[:HAS_FLOOR_LAYOUT]->(f:FloorLayout)
            RETURN f.layout_name AS name, f.total_units_on_floor AS units,
                   f.has_lifts AS lifts, f.has_staircases AS stairs
            """
            floors = list(session.run(cypher_floors, pid=pid))
            print(f"\n--- Floor Layouts ({len(floors)} found) ---")
            for f in floors:
                print(f"  * {f['name']} | Units: {f['units']} | Lifts: {f['lifts']} | Stairs: {f['stairs']}")
                
            # 3. Check Unit Arrays (applicable_buildings)
            cypher_units = """
            MATCH (p:Project {project_id: $pid})-[:HAS_UNIT]->(u:Unit)
            RETURN u.unit_id AS uid, u.applicable_buildings AS bldgs,
                   u.entrance_facing AS facing, u.description AS desc
            LIMIT 2
            """
            units = list(session.run(cypher_units, pid=pid))
            print(f"\n--- Unit Spot Check (showing 2) ---")
            for u in units:
                print(f"  * Unit {u['uid']}")
                print(f"    Facing: {u['facing']}")
                print(f"    Description: {u['desc'][:50]}..." if u['desc'] else "    Description: None")
                print(f"    Applicable Buildings (Array): {u['bldgs']}")

if __name__ == "__main__":
    try:
        verify_new_fields()
    finally:
        driver.close()