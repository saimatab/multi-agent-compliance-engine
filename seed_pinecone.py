import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "compliance-policies"

# Delete the old index if it exists to refresh the dimension structure
if index_name in pc.list_indexes().names():
    print("Deleting old local model index...")
    pc.delete_index(index_name)

print("Creating ultra-lightweight cloud-embedded index...")
pc.create_index(
    name=index_name,
    dimension=1536, # Dimension for text-embedding-3-small
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(index_name)

# Baseline policies
policies = [
    {"id": "p1", "text": "HIPAA rule: All Protected Health Information (PHI) including medical symptoms must be stripped before external transmission."},
    {"id": "p2", "text": "GDPR rule: Credit card numbers, emails, and phone numbers constitute PII and require deterministic masking."}
]

# We upload pure text strings; Pinecone will handle embeddings inside its cloud engine!
vectors_to_upsert = []
for p in policies:
    vectors_to_upsert.append({
        "id": p["id"],
        # Use an empty array structure; we pass data via text metadata
        "values": [0.0] * 1536, 
        "metadata": {"text": p["text"]}
    })

index.upsert(vectors=vectors_to_upsert)
print("Pinecone successfully seeded via Cloud Engine!")
