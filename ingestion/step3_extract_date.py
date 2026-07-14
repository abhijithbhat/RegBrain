"""Step 3: Extract text, detect headers, and find the 'Updated as on' date."""

import re
import sys
import pdfplumber

# ---------------------------------------------------------------------------
# Header patterns (same as step 2)
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

# ---------------------------------------------------------------------------
# Date pattern: "Updated as on <date>"
# Handles formats like:
#   "Updated as on June 12, 2025"
#   "Updated as on 12.06.2025"
#   "Updated as on 12-06-2025"
#   "Updated as on 12/06/2025"
#   "Updated as on December 05, 2024"
# ---------------------------------------------------------------------------
DATE_PATTERN = re.compile(
    r"\(?\s*Updated\s+as\s+on\s+(.+?)[\)\n]",
    re.IGNORECASE,
)


def find_update_date(full_text):
    """Search the full document text for 'Updated as on <date>' and return the date string."""
    match = DATE_PATTERN.search(full_text)
    if match:
        # Clean up: take everything up to the next newline or end of string
        raw_date = match.group(1).strip().split("\n")[0].strip()
        return raw_date
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingestion/step3_extract_date.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    global_line_no = 0
    all_text = []

    print(f"Processing: {pdf_path}\n")

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            all_text.append(text)
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

    # --- Date extraction ---
    full_text = "\n".join(all_text)
    update_date = find_update_date(full_text)

    print(f"\nScanned {page_no} pages, {global_line_no} total lines.")
    print("─" * 60)
    if update_date:
        print(f"📅 Update date found: {update_date}")
    else:
        print("No update date found")


if __name__ == "__main__":
    main()
