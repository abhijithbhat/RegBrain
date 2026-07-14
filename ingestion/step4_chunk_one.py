"""Step 4: Split a PDF into header-delimited chunks and preview them."""

import re
import sys
import pdfplumber

# ---------------------------------------------------------------------------
# Header patterns (same as steps 2–3)
# ---------------------------------------------------------------------------
HEADER_PATTERNS = [
    (r"^Chapter\s+[IVXLCDM]+", "CHAPTER"),
    (r"^Section\s+[IVXLCDM]+", "SECTION"),
    (r"^Part\s+[IVXLCDM]+", "PART"),
    (r"^(?:Annex|Annexure|Appendix)\s+[IVXLCDM\d]+", "ANNEX"),
    (r"^[A-Z]?\d+(?:\.\d+)+", "CLAUSE"),
    (r"^[A-Z]\.\s", "LETTER_CLAUSE"),
]

COMPILED_PATTERNS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in HEADER_PATTERNS]


def match_header(line):
    """Return (label, matched_text) if the line matches a header pattern, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    for pattern, label in COMPILED_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return label, m.group()
    return None


def extract_all_lines(pdf_path):
    """Extract every line from the PDF, tagging each with its page number."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                lines.append((page_no, line))
    return lines


def chunk_by_headers(lines):
    """Split lines into chunks at header boundaries.

    Returns a list of dicts:
        {
            "clause_id": str,      # e.g. "Chapter I", "A.", "4.2.1"
            "label": str,          # e.g. "CHAPTER", "CLAUSE"
            "start_page": int,
            "text": str,           # full text of the chunk
        }
    """
    chunks = []
    current_chunk = None

    for page_no, line in lines:
        result = match_header(line)

        if result:
            # Save the previous chunk (if any)
            if current_chunk is not None:
                chunks.append(current_chunk)

            label, matched_text = result
            current_chunk = {
                "clause_id": matched_text.strip(),
                "label": label,
                "start_page": page_no,
                "text": line + "\n",
            }
        else:
            # Append to current chunk, or start a preamble chunk
            if current_chunk is None:
                current_chunk = {
                    "clause_id": "(preamble)",
                    "label": "PREAMBLE",
                    "start_page": page_no,
                    "text": "",
                }
            current_chunk["text"] += line + "\n"

    # Don't forget the last chunk
    if current_chunk is not None:
        chunks.append(current_chunk)

    return chunks


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingestion/step4_chunk_one.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Processing: {pdf_path}\n")

    lines = extract_all_lines(pdf_path)
    chunks = chunk_by_headers(lines)

    print(f"{'#':>4}  {'Label':<15}  {'Clause ID':<25}  {'Pg':>3}  {'Chars':>6}  First 100 chars")
    print("─" * 120)

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk["text"].replace("\n", " ").strip()[:100]
        print(
            f"{i:>4}  {chunk['label']:<15}  {chunk['clause_id']:<25}  "
            f"{chunk['start_page']:>3}  {len(chunk['text']):>6}  {preview}"
        )

    print(f"\nDone. {len(chunks)} chunks from {lines[-1][0]} pages, {len(lines)} total lines.")


if __name__ == "__main__":
    main()
