"""Final ingestion pipeline: extract, chunk, and export all PDFs in rbi_corpus/."""

import csv
import json
import os
import re
import sys
import pdfplumber

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CORPUS_DIR = "rbi_corpus"
OUTPUT_DIR = "ingestion/chunks"
MANIFEST_PATH = os.path.join(CORPUS_DIR, "manifest.csv")

# ---------------------------------------------------------------------------
# Header patterns (from steps 2-4)
# ---------------------------------------------------------------------------
HEADER_PATTERNS = [
    (r"^Chapter\s+[IVXLCDM]+", "CHAPTER"),
    (r"^Section\s+[IVXLCDM]+", "SECTION"),
    (r"^Part\s+[IVXLCDM]+", "PART"),
    (r"^(?:Annex|Annexure|Appendix)\s+[IVXLCDM\d]+", "ANNEX"),
    (r"^[A-Z]?\d+(?:\.\d+)+", "CLAUSE"),
    (r"^[A-Z]\.\s", "LETTER_CLAUSE"),
]

COMPILED_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), label) for pat, label in HEADER_PATTERNS
]

# Date pattern (from step 3)
DATE_PATTERN = re.compile(
    r"\(?\s*Updated\s+as\s+on\s+(.+?)[\)\n]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers (consolidated from steps 1-4)
# ---------------------------------------------------------------------------
def load_manifest(manifest_path):
    """Load manifest.csv into a dict keyed by filename -> row dict."""
    manifest = {}
    if not os.path.exists(manifest_path):
        return manifest
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "")
            if fname:
                manifest[fname] = row
    return manifest


def match_header(line):
    """Return (label, matched_text) if the line matches a header, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    for pattern, label in COMPILED_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return label, m.group()
    return None


def extract_lines(pdf_path):
    """Extract every line from the PDF, tagged with page number."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                lines.append((page_no, line))
    return lines


def find_update_date(lines):
    """Search all lines for 'Updated as on <date>'."""
    full_text = "\n".join(line for _, line in lines)
    match = DATE_PATTERN.search(full_text)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return None


def chunk_lines(lines):
    """Split lines into chunks at header boundaries.

    Returns list of dicts with keys: clause_id, label, start_page, text.
    """
    chunks = []
    current = None

    for page_no, line in lines:
        result = match_header(line)
        if result:
            if current is not None:
                chunks.append(current)
            label, matched_text = result
            current = {
                "clause_id": matched_text.strip(),
                "label": label,
                "start_page": page_no,
                "text": line + "\n",
            }
        else:
            if current is None:
                current = {
                    "clause_id": "(preamble)",
                    "label": "PREAMBLE",
                    "start_page": page_no,
                    "text": "",
                }
            current["text"] += line + "\n"

    if current is not None:
        chunks.append(current)

    return chunks


def derive_doc_id(filename):
    """Extract a doc_id from the filename (e.g. '148MD' from '148MD.PDF')."""
    base = os.path.splitext(filename)[0]
    return base


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_one(pdf_path, filename, category):
    """Process a single PDF. Returns (doc_id, chunk_list) or raises on error."""
    doc_id = derive_doc_id(filename)

    lines = extract_lines(pdf_path)
    effective_date = find_update_date(lines)
    chunks = chunk_lines(lines)

    output_chunks = []
    for i, chunk in enumerate(chunks):
        clause_id = chunk["clause_id"]

        # Validation: fail loudly if doc_id or clause_id is missing
        if not doc_id:
            raise ValueError(
                f"Missing doc_id for file '{filename}'"
            )
        if not clause_id:
            raise ValueError(
                f"Missing clause_id for chunk #{i+1} in file '{filename}'"
            )

        output_chunks.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "category": category,
            "clause_id": clause_id,
            "clause_label": chunk["label"],
            "start_page": chunk["start_page"],
            "clause_text": chunk["text"].strip(),
            "effective_date": effective_date,
            "source_filename": filename,
        })

    return doc_id, output_chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load manifest for category lookup
    manifest = load_manifest(MANIFEST_PATH)

    # Gather all PDFs
    pdf_files = sorted(
        f for f in os.listdir(CORPUS_DIR)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"No PDF files found in {CORPUS_DIR}/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs in {CORPUS_DIR}/\n")

    docs_ok = 0
    docs_failed = []
    total_chunks = 0

    for filename in pdf_files:
        pdf_path = os.path.join(CORPUS_DIR, filename)

        # Category lookup: try manifest first, else "unknown"
        manifest_row = manifest.get(filename, {})
        category = manifest_row.get("category", "unknown")

        try:
            doc_id, chunks = process_one(pdf_path, filename, category)
            total_chunks += len(chunks)
            docs_ok += 1

            # Write JSON output
            out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            print(f"  ✓ {filename:<60} → {len(chunks):>3} chunks → {out_path}")

        except Exception as e:
            docs_failed.append((filename, str(e)))
            print(f"  ✗ {filename:<60} → ERROR: {e}")

    # --- Summary ---
    print("\n" + "═" * 80)
    print(f"  Docs processed OK : {docs_ok}")
    print(f"  Total chunks      : {total_chunks}")
    print(f"  Docs failed       : {len(docs_failed)}")
    if docs_failed:
        print("\n  Failed documents:")
        for fname, err in docs_failed:
            print(f"    - {fname}: {err}")
    print("═" * 80)


if __name__ == "__main__":
    main()
