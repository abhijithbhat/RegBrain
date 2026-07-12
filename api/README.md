# API

This module exposes the RegBrain Q&A system as a REST service:

- **FastAPI application** — main app with CORS, health-check, and versioned routes
- **Query endpoint** — accepts a user question, orchestrates retrieval → generation → eval
- **Request/response models** — Pydantic schemas for input validation and structured output
- **Configuration** — loads settings from `.env` via `python-dotenv`
