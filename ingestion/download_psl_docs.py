import urllib.request
import re
import csv
import os

headers = {"User-Agent": "Mozilla/5.0"}

def download_pdf_from_page(page_url, filename, title, category):
    req = urllib.request.Request(page_url, headers=headers)
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
    
    # Look for PDF links
    pdf_match = re.search(r"href=[\"\']?(https?://rbidocs\.rbi\.org\.in/rdocs/[^\"\' >]+\.pdf)[\"\']?", html, re.IGNORECASE)
    if not pdf_match:
        pdf_match = re.search(r"href=[\"\']?([^\"\' >]+\.pdf)[\"\']?", html, re.IGNORECASE)
    
    if pdf_match:
        pdf_url = pdf_match.group(1)
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.rbi.org.in/Scripts/" + pdf_url
        print(f"Downloading {title} from {pdf_url} ...")
        
        pdf_req = urllib.request.Request(pdf_url, headers=headers)
        dest_path = os.path.join("ingestion", "rbi_corpus", filename)
        with urllib.request.urlopen(pdf_req) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        print(f"Saved {dest_path} ({os.path.getsize(dest_path)} bytes)")
        
        # Append to manifest.csv if not already present
        manifest_path = os.path.join("ingestion", "rbi_corpus", "manifest.csv")
        existing_lines = open(manifest_path, "r", encoding="utf-8").read() if os.path.exists(manifest_path) else ""
        if filename not in existing_lines:
            with open(manifest_path, "a", encoding="utf-8", newline="") as mf:
                writer = csv.writer(mf)
                writer.writerow([filename, category, title])
            print(f"Appended {filename} to manifest.csv")
    else:
        print(f"ERROR: Could not find PDF link on {page_url}")

# 1. 2020 PSL MD
download_pdf_from_page(
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11959",
    "PSL_2020_MD.PDF",
    "Master Direction - Priority Sector Lending (PSL) – Targets and Classification (2020)",
    "Commercial_Banks"
)

# 2. 2025 PSL MD
download_pdf_from_page(
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12799",
    "PSL_2025_MD.PDF",
    "Master Directions - Reserve Bank of India (Priority Sector Lending – Targets and Classification) Directions, 2025",
    "Commercial_Banks"
)
