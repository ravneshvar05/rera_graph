"""
Simulate what happens in the Streamlit app: load user_settings.json and test
generate_cypher with those keys, same as the app does.
"""
import os, json
os.chdir(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except Exception:
    pass

# Load user_settings.json exactly like the app does
with open("user_settings.json") as f:
    user_settings = json.load(f)

print("=== user_settings keys ===")
for k, v in user_settings.items():
    display = v[:30] + "..." if len(v) > 30 else v
    print(f"  {k}: {display!r}")

# Build the pools to see which keys are detected
from graphrag_config import build_gemini_pool, build_groq_pool

gemini_pool = build_gemini_pool(user_settings)
groq_pool   = build_groq_pool(user_settings)

print(f"\nGemini pool size: {len(gemini_pool)}")
print(f"Groq pool size: {len(groq_pool)}")

# Now test generate_cypher with the full user_settings (as app does)
print("\n=== generate_cypher with user_settings ===")
from graphrag_cypher import generate_cypher

TEST_QUERY = "find me house in sarkhej and vinzol"
cq = generate_cypher(TEST_QUERY, api_keys=user_settings)

print(f"  Engine: {cq.engine_used}")
print(f"  Query type: {cq.query_type}")
print(f"  Intent neighbourhood: {cq.intent.neighbourhood}")
print(f"  Intent city: {cq.intent.city}")
print(f"  Vector query: {cq.vector_query}")
print(f"  Params: {cq.params}")
