from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

COLLECTION = "chunks_openai_semantic_large"  # change as needed

points, _ = client.scroll(
    collection_name=COLLECTION,
    limit=1,
    with_payload=True,
    with_vectors=False,
)

p = points[0]

print("Point ID:", p.id)
print("Payload keys:", sorted(p.payload.keys()))
print("Payload:", p.payload)
