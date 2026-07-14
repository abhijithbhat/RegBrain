"""Step 2: Extract text from a PDF and detect RBI header/clause patterns."""

import re
import sys
import pdfplumber

# Patterns for common RBI document headers
HEADER_PATTERNS = [
    # "Chapter I", "Chapter IV", "Chapter XIV" etc.
    (r"^Chapter\s+[IVXLCDM]+", "CHAPTER"),
    # "Section I", "Section IV" etc.
    (r"^Section\s+[IVXLCDM]+", "SECTION"),
    # "Part I", "Part IV" etc.
    (r"^Part\s+[IVXLCDM]+", "PART"),
    # "Annex I", "Appendix II" etc.
    (r"^(?:Annex|Annexure|Appendix)\s+[IVXLCDM\d]+", "ANNEX"),
    # Sub-clause numbering: "4.2", "4.2.1", "B.11.2" etc. at start of line
    (r"^[A-Z]?\d+(?:\.\d+)+", "CLAUSE"),
    # Lettered sub-clauses: "A.", "B.", "(a)", "(i)" at start of line
    (r"^[A-Z]\.\s", "LETTER_CLAUSE"),
]

# Compile all patterns
COMPILED_PATTERNS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in HEADER_PATTERNS]


def detect_headers(text_lines):
    """Yield (line_number, pattern_type, matched_text, full_line) for each header match."""
    for line_no, line in enumerate(text_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, label in COMPILED_PATTERNS:
            match = pattern.match(stripped)
            if match:
                yield line_no, label, match.group(), stripped
                break  # one match per line is enough


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingestion/step2_detect_headers.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    global_line_no = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = text.split("\n")

            for local_line_no, line in enumerate(lines, start=1):
                global_line_no += 1
                stripped = line.strip()
                if not stripped:
                    continue

                for pattern, label in COMPILED_PATTERNS:
                    match = pattern.match(stripped)
                    if match:
                        print(
                            f"[Page {page_no:>3}, Line {global_line_no:>5}]  "
                            f"{label:<15}  {match.group():<20}  │ {stripped[:100]}"
                        )
                        break

    print(f"\nDone. Scanned {page_no} pages, {global_line_no} total lines.")


if __name__ == "__main__":
    main()
