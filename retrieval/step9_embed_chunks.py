"""Step 9: Embed all chunk texts from ingestion/chunks/ using bge-small-en-v1.5."""

import json
import os
import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNKS_DIR = "ingestion/chunks"
BATCH_SIZE = 32
PROGRESS_EVERY = 20


def load_all_chunks(chunks_dir):
    """Load every chunk from all JSON files in the chunks directory."""
    all_chunks = []
    for filename in sorted(os.listdir(chunks_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, filename), encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks


def main():
    print(f"Loading chunks from {CHUNKS_DIR}/ ...")
    chunks = load_all_chunks(CHUNKS_DIR)
    texts = [c["clause_text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks\n")

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} chunks (batch_size={BATCH_SIZE}) ...\n")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )

    # Print progress summary (since encode runs in one call,
    # we report the final result)
    elapsed = time.time() - start

    print(f"✅ Embedded {len(embeddings)} / {len(texts)} chunks")
    print(f"   Embedding dimension : {embeddings.shape[1]}")
    print(f"   Total time          : {elapsed:.1f}s")
    print(f"   Speed               : {len(embeddings) / elapsed:.0f} chunks/sec")


if __name__ == "__main__":
    main()
