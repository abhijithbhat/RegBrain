"""Step 1: Extract raw text from a single PDF, page by page."""

import sys
import pdfplumber


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingestion/step1_extract_one.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            print(f"--- PAGE {i} ---")
            print(text)
            print()


if __name__ == "__main__":
    main()
