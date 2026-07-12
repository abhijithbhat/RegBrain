# Retrieval

This module implements hybrid search over the indexed regulatory corpus:

- **Dense retrieval** — semantic similarity search against Qdrant vectors
- **Sparse retrieval** — BM25 keyword matching via `rank_bm25`
- **Fusion** — combine and re-rank results from both retrievers (e.g., Reciprocal Rank Fusion)
- **Context assembly** — prepare the top-k passages with metadata for the generation stage
