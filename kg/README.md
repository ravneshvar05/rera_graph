# RERA Knowledge Graph — `kg/`

A production-grade **Neo4j Knowledge Graph + ChromaDB vector layer** built from real-estate brochure JSON files.

## Folder Structure

```
kg/
├── kg_config.py         # Settings (reads .env)
├── kg_schema.py         # Create Neo4j constraints + indexes (run once)
├── kg_ingest.py         # Main pipeline: JSON → Neo4j graph + ChromaDB vectors
├── kg_geo_enrich.py     # Post-ingest: Zone hierarchy + landmark aliases
├── requirements.txt
├── .env.example         # Copy → .env and fill in your values
└── .gitignore
```

## Quick Start

### 1. Prerequisites

- **Neo4j 5** running locally:
  ```bash
  docker run -p 7687:7687 -p 7474:7474 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:5
  ```
- Python 3.11+

### 2. Setup

```bash
# Inside this kg/ directory:
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

cp .env.example .env
# Edit .env — fill in NEO4J_PASSWORD and GEMINI_API_KEY
```

### 3. Run (in order)

```bash
# Step 1 — Create Neo4j schema (run once)
python kg_schema.py

# Step 2 — Ingest all JSONs from ../output/
python kg_ingest.py

# Step 3 — Wire zone hierarchy + aliases (run once after ingest)
python kg_geo_enrich.py
```

### CLI Options

```bash
python kg_ingest.py --help

  --force        Wipe graph + ChromaDB and re-ingest everything
  --dry-run      Validate JSONs, print what would be ingested, no writes
  --workers N    Thread count for parallel ingestion (default: 8)
  --dir PATH     Override JSON directory

python kg_geo_enrich.py --reset   # Wipe PART_OF/SAME_AS/ALIAS_OF edges and re-run
```

## Architecture

```
JSON files (output/)
       │
       ▼
kg_ingest.py
  ├─ Neo4j Graph ─────────────────────────────────────────────────
  │    Project ─[:LOCATED_IN]──► Neighbourhood ─[:PART_OF]──► Zone ─[:IN_CITY]──► City
  │    Project ─[:BUILT_BY]────► Developer
  │    Project ─[:HAS_UNIT]────► Unit ─[:HAS_ROOM]──► Room
  │    Project ─[:NEAR]────────► Landmark ─[:SAME_AS]──► Neighbourhood
  │    Project ─[:HAS_AMENITY]─► Amenity
  │    Neighbourhood ─[:ALIAS_OF]─► Neighbourhood
  │
  └─ ChromaDB Vectors ─────────────────────────────────────────────
       One document per Unit — combines:
         • Project name, developer, address, RERA, status
         • Society description + facilities + building names
         • Floor layout info (floors, lifts)
         • Unit description, area, facing, applicable buildings
         • ALL room details (name, type, dimensions, floor, bath attachment)
         • ALL amenity strings (full raw text)
         • ALL nearby landmarks with type classification
```

## Scaling to 20,000 JSONs

| Entity | Estimated count |
|---|---|
| `:Project` | ~20,000 |
| `:Unit` | ~80,000 |
| `:Room` | ~400,000 |
| `:Landmark` | ~2,000 (de-duplicated via MERGE) |
| `:Amenity` | ~500 (de-duplicated via MERGE) |
| ChromaDB docs | ~80,000 (one per unit) |

**Expected ingestion time at 20k files:**  
~45 min with 8 workers (CPU bound on embedding generation).  
~25 min with 16 workers on a high-core machine.

## Next Phase (Chatbot Integration)

After the graph is built, wire it into the chatbot:
- `kg_query_planner.py` — LLM extracts Intent from user query
- `kg_search.py` — Cypher traversal + ChromaDB query + merge/rank
- `kg_app.py` — Streamlit UI using the new pipeline
