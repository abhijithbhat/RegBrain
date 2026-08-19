"""Unified index builder: dense (Qdrant) + sparse (BM25) in one pass."""

import json
import os
import pickle
import time
import traceback
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNKS_DIR = "ingestion/chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BM25_PATH = "retrieval/bm25_index.pkl"

COLLECTION = "regbrain"
VECTOR_SIZE = 384
EMBED_BATCH = 64
UPLOAD_BATCH = 20
MAX_RETRIES = 3

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

PAYLOAD_FIELDS = [
    "doc_id", "chunk_index", "category", "clause_id", "clause_label", "start_page",
    "effective_date", "source_filename", "clause_text",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_all_chunks(chunks_dir):
    """Load every chunk from all JSON files."""
    all_chunks = []
    for filename in sorted(os.listdir(chunks_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, filename), encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── 1. Load chunks ─────────────────────────────────────────────────
    print(f"Loading chunks from {CHUNKS_DIR}/ ...")
    chunks = load_all_chunks(CHUNKS_DIR)
    texts = [c["clause_text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks\n")

    # ── 2. Embed ───────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} chunks (batch_size={EMBED_BATCH}) ...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=True)
    t_embed = time.time() - t0
    print(f"Embedding done in {t_embed:.1f}s ({len(texts)/t_embed:.0f} chunks/sec)\n")

    # ── 3. Upload to Qdrant ────────────────────────────────────────────
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    if client.collection_exists(COLLECTION):
        print(f"Recreating collection '{COLLECTION}' from scratch ...")
        client.delete_collection(COLLECTION)

    print(f"Creating collection '{COLLECTION}' (size={VECTOR_SIZE}, cosine) ...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Build points
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = {field: chunk.get(field) for field in PAYLOAD_FIELDS}
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload=payload,
        ))

    # Upload in batches with retry
    print(f"\nUploading {len(points)} points to Qdrant (batch_size={UPLOAD_BATCH}) ...")
    t0 = time.time()
    uploaded = 0

    for batch_start in range(0, len(points), UPLOAD_BATCH):
        batch = points[batch_start : batch_start + UPLOAD_BATCH]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=COLLECTION, points=batch)
                break
            except Exception:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"  ⚠ Batch at {batch_start} failed (attempt {attempt}), retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  ✗ Batch at {batch_start} failed after {MAX_RETRIES} attempts")
                    traceback.print_exc()
                    raise

        uploaded += len(batch)
        if uploaded % 100 == 0 or uploaded == len(points):
            print(f"  Uploaded {uploaded:>5} / {len(points)} chunks")

    t_upload = time.time() - t0
    qdrant_count = client.get_collection(COLLECTION).points_count

    # ── 4. Build BM25 index ────────────────────────────────────────────
    print(f"\nBuilding BM25 index ...")
    corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    bm25_size_mb = os.path.getsize(BM25_PATH) / (1024 * 1024)

    # ── 5. Summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Total chunks processed  : {len(chunks)}")
    print(f"  Qdrant collection       : {COLLECTION} ({qdrant_count} points)")
    print(f"  BM25 index              : {BM25_PATH} ({bm25_size_mb:.1f} MB)")
    print(f"  Embed time              : {t_embed:.1f}s")
    print(f"  Upload time             : {t_upload:.1f}s")
    print(f"{'═' * 60}")
    print("✅ Both indexes built!")


if __name__ == "__main__":
    main()
