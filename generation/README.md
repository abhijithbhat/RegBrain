# Generation

This module handles LLM-powered answer generation:

- **Prompt construction** — build grounded prompts from retrieved passages with citation instructions
- **LLM integration** — call the language model (e.g., Gemini, GPT) to produce answers
- **Citation grounding** — ensure every claim in the response maps back to a source passage
- **Post-processing** — format the answer with inline references and confidence indicators
