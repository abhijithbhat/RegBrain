# RegBrain

RegBrain is a self-verifying Retrieval-Augmented Generation (RAG) system designed for RBI (Reserve Bank of India) regulatory Q&A. It ingests RBI circulars and master directions as PDFs, chunks and embeds them into a hybrid vector + BM25 index, retrieves relevant passages for a user query, generates grounded answers with citation, and then automatically evaluates response quality using the RAGAS framework — closing the loop between generation and verification.

## Project Structure

```
ingestion/   – PDF parsing, chunking, and embedding pipeline
retrieval/   – Hybrid search (dense + BM25) over Qdrant
generation/  – LLM answer generation with citation grounding
api/         – FastAPI service exposing the Q&A endpoint
frontend/    – Lightweight web UI for asking questions
eval/        – RAGAS-based evaluation and self-verification
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/abhijithbhat/RegBrain.git
cd RegBrain

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```
