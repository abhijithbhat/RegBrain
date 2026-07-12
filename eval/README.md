# Eval

This module implements self-verification using the RAGAS framework:

- **Faithfulness** — does the answer stay true to the retrieved passages?
- **Answer relevancy** — is the response actually addressing the user's question?
- **Context precision / recall** — are the retrieved chunks the right ones?
- **End-to-end scoring** — aggregate metrics to flag low-confidence answers for review
- **Test datasets** — curated Q&A pairs for regression testing the full pipeline
