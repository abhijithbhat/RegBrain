"""Step 10: Test connection to Qdrant Cloud — create, insert, read, cleanup."""

import os
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "connection_test"
VECTOR_SIZE = 384


def main():
    # --- Connect ---
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # --- Create collection ---
    print(f"Creating collection '{COLLECTION}' (size={VECTOR_SIZE}, cosine) ...")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # --- Upload one dummy point ---
    dummy_vector = [random.random() for _ in range(VECTOR_SIZE)]
    point = PointStruct(
        id=1,
        vector=dummy_vector,
        payload={"note": "hello from RegBrain"},
    )
    print("Uploading 1 dummy point ...")
    client.upsert(collection_name=COLLECTION, points=[point])

    # --- Retrieve it back ---
    print("Retrieving point ...")
    results = client.retrieve(collection_name=COLLECTION, ids=[1])
    for r in results:
        print(f"  id      : {r.id}")
        vector_val = r.vector
        if isinstance(vector_val, list):
            print(f"  vector  : [{vector_val[:5]}...] (first 5 of {VECTOR_SIZE})")
        else:
            print("  vector  : (not returned)")

    # --- Cleanup ---
    print(f"Deleting collection '{COLLECTION}' ...")
    client.delete_collection(collection_name=COLLECTION)

    print("\n✅ Qdrant connection test passed!")


if __name__ == "__main__":
    main()
