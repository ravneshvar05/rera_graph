# GraphRAG — Property Recommendation System

A hybrid **GraphRAG + Vector Search** chatbot for residential real estate. Users describe what they want in natural language; the system searches a Neo4j knowledge graph and a ChromaDB vector store and generates a recommendation using Gemini.

---

## Architecture

```
User Query (natural language)
       ↓
graphrag_intent.py    ← Gemini parses query → QueryIntent (BHK, location, amenities…)
       ↓
graphrag_retriever.py ← Dual retrieval:
                         • Neo4j Cypher  (precise structural filtering)
                         • ChromaDB      (semantic / fuzzy matching)
                       ← Results merged & deduplicated
       ↓
graphrag_answer.py    ← Gemini synthesises final recommendation
       ↓
graphrag_app.py       ← Streamlit chat UI
```

**Data source:** `../output/` — brochure JSON files processed by the `kg/` pipeline (Neo4j + ChromaDB already populated).

---

## Quick Start

### Prerequisites
- Neo4j running (see `kg/README.md` for Docker command)
- KG pipeline already ingested (`kg/kg_ingest.py` run)

### Setup

```powershell
cd graphrag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy and fill your `.env`:
```powershell
copy .env.example .env
# Edit .env — set NEO4J_PASSWORD
```

### Run

```powershell
streamlit run graphrag_app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Files

| File | Purpose |
|---|---|
| `graphrag_config.py` | Configuration (env vars, paths) |
| `graphrag_intent.py` | LLM query → structured `QueryIntent` |
| `graphrag_retriever.py` | Dual Neo4j + ChromaDB retrieval, context assembly |
| `graphrag_answer.py` | LLM-based recommendation generator |
| `graphrag_app.py` | Streamlit chat interface |

---

## Example Queries

| Query | What happens |
|---|---|
| `"Show me all available projects"` | Graph returns all 20 projects |
| `"2 BHK in Vinzol with pool"` | Cypher filters by neighbourhood + amenity; vector adds semantic matches |
| `"Luxury villa near airport"` | Vector drives discovery; graph grounds with landmark hop |
| `"1 BHK near school for family"` | Landmark type=EDUCATION + semantic keywords "family" |
| `"3 BHK with gym and clubhouse under 80L"` | Amenity filtering via graph + semantic match via vector |

---

## Tuning

Edit `.env` to adjust:
- `VECTOR_TOP_K` — how many ChromaDB hits to retrieve (default: 15)
- `GRAPH_MAX_RESULTS` — max Neo4j results per query (default: 30)
- `FINAL_TOP_N` — max projects shown to user (default: 10)
