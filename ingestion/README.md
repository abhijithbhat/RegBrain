# Ingestion

This module handles the PDF ingestion pipeline:

- **PDF parsing** — extract text and tables from RBI circulars/master directions using `pdfplumber`
- **Chunking** — split documents into semantically meaningful passages with metadata (source, page, date)
- **Embedding** — generate dense vector embeddings via `sentence-transformers`
- **Indexing** — upsert chunks into the Qdrant vector store and build a BM25 index
