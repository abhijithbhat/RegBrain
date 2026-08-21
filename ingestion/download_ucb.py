"""Download RBI Urban Co-operative Banks (UCB) Master Directions PDFs with session cookies."""

import csv
import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_INDEX_URL = "https://rbi.org.in/scripts/BS_ViewMasterDirections.aspx"
CORPUS_DIR = "ingestion/rbi_corpus"
MANIFEST_PATH = os.path.join(CORPUS_DIR, "manifest.csv")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Referer": "https://rbi.org.in/scripts/BS_ViewMasterDirections.aspx",
})


def init_session():
    print("Initializing session on RBI Master Directions index...")
    session.get(BASE_INDEX_URL, timeout=30)
    print(f"Session cookies established: {len(session.cookies)} cookies")


def get_ucb_links():
    resp = session.get(BASE_INDEX_URL, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        text = a.text.strip()
        href = a["href"]
        if ("Urban Co-operative" in text or "Urban Cooperative" in text or "UCB" in text) and href.startswith("BS_ViewMasDirections.aspx?id="):
            detail_url = "https://rbi.org.in/scripts/" + href
            links.append((text, detail_url))
    return links


def get_pdf_url_from_detail(detail_url):
    try:
        resp = session.get(detail_url, headers={"Referer": BASE_INDEX_URL}, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "notification/PDFs/" in href and href.lower().endswith(".pdf"):
                return href if href.startswith("http") else "https://rbidocs.rbi.org.in" + href
            if "/content/pdfs/" in href and href.lower().endswith(".pdf") and not any(x in href.lower() for x in ["_an", "utkarsh", "access"]):
                return href if href.startswith("http") else "https://rbidocs.rbi.org.in" + href
    except Exception as e:
        print(f"  Error fetching detail page {detail_url}: {e}")
    return None


def main():
    init_session()
    ucb_links = get_ucb_links()
    print(f"Found {len(ucb_links)} UCB Master Directions on RBI index.")

    # Load existing manifest
    manifest_rows = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))

    # Clean existing non-PDF files from UCB downloads
    for row in list(manifest_rows):
        if row.get("category") == "UCB":
            fpath = os.path.join(CORPUS_DIR, row["filename"])
            if os.path.exists(fpath):
                with open(fpath, "rb") as check_f:
                    if not check_f.read(10).startswith(b"%PDF"):
                        print(f"Removing invalid HTML file: {row['filename']}")
                        os.remove(fpath)
                        manifest_rows.remove(row)

    existing_filenames = {row["filename"].upper() for row in manifest_rows}
    downloaded = 0

    for title, detail_url in ucb_links:
        pdf_url = get_pdf_url_from_detail(detail_url)
        if not pdf_url:
            print(f"  [SKIPPED - No PDF link found]: {title}")
            continue

        raw_fname = os.path.basename(pdf_url)
        filename = raw_fname.upper()
        filepath = os.path.join(CORPUS_DIR, raw_fname)

        if filename in existing_filenames and os.path.exists(filepath):
            with open(filepath, "rb") as check_f:
                if check_f.read(10).startswith(b"%PDF"):
                    print(f"  [VALID PDF EXISTS]: {raw_fname} -> {title[:50]}")
                    continue

        print(f"  Downloading clean PDF: {raw_fname} ({title[:50]}...)")
        try:
            r = session.get(pdf_url, headers={"Referer": detail_url}, timeout=60)
            if r.status_code == 200 and r.content.startswith(b"%PDF"):
                with open(filepath, "wb") as f:
                    f.write(r.content)
                manifest_rows.append({
                    "filename": raw_fname,
                    "category": "UCB",
                    "title": title
                })
                existing_filenames.add(filename)
                downloaded += 1
                print(f"    ✓ Valid PDF saved: {raw_fname} ({len(r.content):,} bytes)")
            else:
                print(f"    ✗ Download failed or not PDF (status={r.status_code}, header={r.content[:15]})")
        except Exception as e:
            print(f"    ✗ Download error for {raw_fname}: {e}")

        time.sleep(0.5)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "title"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nCompleted! Downloaded {downloaded} clean UCB PDFs. Total files in manifest: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
