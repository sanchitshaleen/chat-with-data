import sys
sys.path.insert(0, "./server")

# Connect to Qdrant directly
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize client and embeddings
client = QdrantClient(host="localhost", port=6335)
embeddings_model = SentenceTransformer("BAAI/bge-m3-embed-large", model_kwargs={"trust_remote_code":True})

# Test query
query = "what is the recommended dosage for ibrance"
query_embedding = embeddings_model.encode(query).tolist()

# Search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)

print(f"\n✅ Retrieved {len(results)} documents for query: '{query}'\n")

for i, result in enumerate(results):
    print(f"Doc {i+1} (Score: {result.score:.4f}):")
    payload = result.payload
    print(f"  Source: {payload.get('source', 'N/A')}")
    print(f"  Page: {payload.get('page', 'N/A')}")
    print(f"  Content preview: {payload.get('page_content', '')[:150]}...")
    print()
