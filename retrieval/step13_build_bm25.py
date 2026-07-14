"""Step 13: Build a BM25 keyword index over all chunks and save to disk."""

import json
import os
import pickle
from rank_bm25 import BM25Okapi

CHUNKS_DIR = "ingestion/chunks"
OUTPUT_PATH = "retrieval/bm25_index.pkl"


def load_all_chunks(chunks_dir):
    """Load every chunk from all JSON files."""
    all_chunks = []
    for filename in sorted(os.listdir(chunks_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, filename), encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    return all_chunks


def tokenize(text):
    """Simple whitespace tokenizer with lowercasing."""
    return text.lower().split()


def main():
    print(f"Loading chunks from {CHUNKS_DIR}/ ...")
    chunks = load_all_chunks(CHUNKS_DIR)
    print(f"Loaded {len(chunks)} chunks\n")

    # Tokenize each chunk's clause_text
    print("Tokenizing ...")
    corpus = [tokenize(c["clause_text"]) for c in chunks]

    # Build BM25 index
    print("Building BM25 index ...")
    bm25 = BM25Okapi(corpus)

    # Save index + metadata
    print(f"Saving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n✅ BM25 index built!")
    print(f"   Total chunks indexed : {len(chunks)}")
    print(f"   Index file           : {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
