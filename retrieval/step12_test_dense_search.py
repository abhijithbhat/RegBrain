"""Step 12: Dense semantic search against the 'regbrain' Qdrant collection."""

import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "regbrain"
TOP_K = 5

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieval/step12_test_dense_search.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')

    # --- Embed the query ---
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(query).tolist()

    # --- Search Qdrant ---
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
    )

    # --- Print results ---
    print(f"Top {TOP_K} results:\n")
    print(f"{'#':>2}  {'Score':>6}  {'Doc ID':<15}  {'Clause':<15}  {'Category':<18}  Text (first 150 chars)")
    print("─" * 130)

    for i, hit in enumerate(results.points, start=1):
        p = hit.payload if hit.payload is not None else {}
        text_preview = p.get("clause_text", "")[:150].replace("\n", " ")
        print(
            f"{i:>2}  {hit.score:>6.4f}  "
            f"{p.get('doc_id', '?'):<15}  "
            f"{p.get('clause_id', '?'):<15}  "
            f"{p.get('category', '?'):<18}  "
            f"{text_preview}"
        )

    print()


if __name__ == "__main__":
    main()
