import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize free Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "compliance-policies"

# Create a serverless index on the free tier if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating free Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Dimension of the free all-MiniLM-L6-v2 model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize local, completely free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Baseline compliance rules
policies = [
    {"id": "p1", "text": "HIPAA rule: All Protected Health Information (PHI) including medical symptoms must be stripped before external transmission."},
    {"id": "p2", "text": "GDPR rule: Credit card numbers, emails, and phone numbers constitute PII and require deterministic masking."}
]

# Embed and upload
vectors_to_upsert = []
for p in policies:
    vector_values = model.encode(p["text"]).tolist()
    vectors_to_upsert.append({
        "id": p["id"],
        "values": vector_values,
        "metadata": {"text": p["text"]}
    })

index.upsert(vectors=vectors_to_upsert)
print("Pinecone vector database successfully seeded on the free tier!")
