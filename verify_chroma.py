import os
import chromadb
from dotenv import load_dotenv
import json

# Adjust this path if needed
load_dotenv('d:/Rerachatbot/kg/.env')

CHROMA_PATH = os.getenv("CHROMA_PATH", "d:/Rerachatbot/kg/chroma_db")
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
        
    print("\n--- Fetching a Sample Document ---")
    
    # We will query specifically for "OumOrbit_Brochure_1__0"
    results = col.get(ids=["OumOrbit_Brochure_1__0"], include=["documents", "metadatas"])
    
    if not results['documents']:
        print("Could not find document OumOrbit_Brochure_1__0. Fetching a random one instead...")
        results = col.get(limit=1, include=["documents", "metadatas"])
        
    doc = results['documents'][0]
    meta = results['metadatas'][0]
    doc_id = results['ids'][0]
    
    print(f"\nID: {doc_id}")
    print(f"\nMETADATA:\n{json.dumps(meta, indent=2)}")
    print(f"\nEXACT EMBEDDED TEXT (This is what the vector represents):\n")
    print("-" * 80)
    print(doc)
    print("-" * 80)
    
if __name__ == "__main__":
    verify_chroma()
