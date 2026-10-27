import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "paste_your_key_here":
    raise SystemExit("ERROR: GEMINI_API_KEY is missing in your .env file.")

gen_model = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite")
embed_model = os.getenv("EMBED_MODEL", "gemini-embedding-001")

client = genai.Client(api_key=api_key)

# Test 1: text generation
resp = client.models.generate_content(
    model=gen_model,
    contents="Reply with exactly: Gemini connection OK",
)
print("Generation test:", resp.text.strip())

# Test 2: embeddings
emb = client.models.embed_content(
    model=embed_model,
    contents=["What documents are needed to apply for a scholarship?"],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    ),
)
print("Embedding test: vector length =", len(emb.embeddings[0].values))