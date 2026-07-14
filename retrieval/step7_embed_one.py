"""Step 7: Load a sentence-transformer model and embed a single example text."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXAMPLE_TEXT = "NBFCs must maintain a minimum capital adequacy ratio."


def main():
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding: \"{EXAMPLE_TEXT}\"")
    vector = model.encode(EXAMPLE_TEXT)

    print(f"\nVector length : {len(vector)}")
    print(f"First 10 nums: {vector[:10].tolist()}")


if __name__ == "__main__":
    main()
