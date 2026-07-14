"""Step 8: Embed multiple sentences and compare their cosine similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

SENTENCE_A = "NBFCs must maintain a minimum capital adequacy ratio."
SENTENCE_B = "Non-banking financial companies need sufficient capital reserves."
SENTENCE_C = "The bank's branch will be closed on public holidays."


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding sentences ...\n")
    vec_a = model.encode(SENTENCE_A)
    vec_b = model.encode(SENTENCE_B)
    vec_c = model.encode(SENTENCE_C)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    print(f'A: "{SENTENCE_A}"')
    print(f'B: "{SENTENCE_B}"')
    print(f'C: "{SENTENCE_C}"')
    print()
    print(f"Similarity A ↔ B (related)  : {sim_ab:.4f}")
    print(f"Similarity A ↔ C (unrelated): {sim_ac:.4f}")


if __name__ == "__main__":
    main()
