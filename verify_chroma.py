"""Verify ChromaDB dual-document embedding after re-ingestion."""
import os
import chromadb
from dotenv import load_dotenv
import json

load_dotenv('d:/Rerachatbot/kg/.env')

CHROMA_PATH = os.getenv("CHROMA_PATH", "d:/Rerachatbot/kg/db/chroma_kg")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "rera_kg_units")

def verify_chroma():
    print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    try:
        col = client.get_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return
        
    count = col.count()
    print(f"Total documents in ChromaDB collection '{CHROMA_COLLECTION}': {count}")
    
    if count == 0:
        print("ChromaDB is empty!")
        return

    # Count doc types
    try:
        proj_docs = col.get(where={"doc_type": {"$eq": "project"}}, include=["metadatas"])
        unit_docs = col.get(where={"doc_type": {"$eq": "unit"}}, include=["metadatas"])
        print(f"\n  Project-level docs: {len(proj_docs['ids'])}")
        print(f"  Unit-level docs:    {len(unit_docs['ids'])}")
    except Exception as e:
        print(f"  Could not filter by doc_type: {e}")

    # Show a project doc
    print("\n" + "="*80)
    print("SAMPLE PROJECT-LEVEL DOCUMENT")
    print("="*80)
    try:
        res = col.get(where={"doc_type": {"$eq": "project"}}, limit=1, include=["documents", "metadatas"])
        if res['documents']:
            print(f"\nID: {res['ids'][0]}")
            print(f"\nMETADATA:\n{json.dumps(res['metadatas'][0], indent=2)}")
            print(f"\nEMBEDDED TEXT (first 800 chars):\n")
            print("-"*80)
            print(res['documents'][0][:800])
            print("-"*80)
    except Exception as e:
        print(f"Error: {e}")

    # Show a unit doc
    print("\n" + "="*80)
    print("SAMPLE UNIT-LEVEL DOCUMENT")
    print("="*80)
    try:
        res = col.get(where={"doc_type": {"$eq": "unit"}}, limit=1, include=["documents", "metadatas"])
        if res['documents']:
            print(f"\nID: {res['ids'][0]}")
            print(f"\nMETADATA:\n{json.dumps(res['metadatas'][0], indent=2)}")
            print(f"\nEMBEDDED TEXT (first 800 chars):\n")
            print("-"*80)
            print(res['documents'][0][:800])
            print("-"*80)
    except Exception as e:
        print(f"Error: {e}")

    # Check new metadata fields
    print("\n" + "="*80)
    print("METADATA FIELD COVERAGE CHECK")
    print("="*80)
    try:
        sample = col.get(limit=10, include=["metadatas"])
        expected_fields = ['doc_type', 'project_id', 'project_name', 'city', 'neighbourhood',
                          'developer', 'bhk', 'property_type', 'area_sqft', 'has_clubhouse',
                          'has_pool', 'has_park', 'has_parking', 'project_status',
                          'entrance_facing', 'room_names', 'amenities']
        for field in expected_fields:
            present = sum(1 for m in sample['metadatas'] if field in m)
            print(f"  {field:25s}: {present}/10 docs have this field")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_chroma()
