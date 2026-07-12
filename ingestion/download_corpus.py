import requests
from bs4 import BeautifulSoup
import os, csv, time

CATEGORIES = {
    411: "NBFC",
    403: "Commercial_Banks",
    404: "Small_Finance_Banks",
    405: "Payments_Banks",
}

BASE_URL = "https://rbi.org.in/scripts/BS_ViewMasterDirections.aspx?did={}"
OUTPUT_DIR = "rbi_corpus"
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
manifest_rows = []
seen_urls = set()

for did, category_name in CATEGORIES.items():
    url = BASE_URL.format(did)
    print(f"Fetching category: {category_name}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_links = [
        a for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".pdf") and "rbidocs.rbi.org.in" in a["href"].lower()
    ]
    print(f"  Found {len(pdf_links)} actual document links")

    for link in pdf_links:
        pdf_url = link["href"]
        if not pdf_url.startswith("http"):
            pdf_url = "https://rbidocs.rbi.org.in" + pdf_url

        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)

        # get the real title: try link text, then image alt text, then the URL itself
        title = link.text.strip()
        if not title:
            img = link.find("img")
            if img and img.get("alt"):
                title = img["alt"].strip()
        if not title:
            title = os.path.basename(pdf_url)

        # strip the leading "PDF - " prefix RBI adds to alt text, if present
        if title.lower().startswith("pdf - "):
            title = title[6:]

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:100]
        doc_id = os.path.basename(pdf_url).split(".")[0]  # unique id from the file itself
        filename = f"{category_name}__{safe_title}__{doc_id}.pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            continue

        try:
            pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
            with open(filepath, "wb") as f:
                f.write(pdf_resp.content)
            manifest_rows.append({
                "category": category_name,
                "title": title,
                "source_url": pdf_url,
                "filename": filename
            })
            print(f"  Downloaded: {filename}")
        except Exception as e:
            print(f"  FAILED: {title} -> {e}")

        time.sleep(1)

with open(os.path.join(OUTPUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "title", "source_url", "filename"])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nDone. Downloaded {len(manifest_rows)} PDFs into '{OUTPUT_DIR}/'. Manifest: manifest.csv")