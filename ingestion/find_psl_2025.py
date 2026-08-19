import urllib.request
import re

url = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

clean_html = re.sub(r"\s+", " ", html)
for m in re.finditer(r"<a[^>]*href=[\"\']?([^\"\' >]+)[\"\']?[^>]*>(.*?)</a>", clean_html, re.IGNORECASE):
    href, text = m.group(1), m.group(2)
    text_clean = re.sub(r"<[^>]+>", "", text).strip()
    if "priority" in text_clean.lower() or "financial inclusion" in text_clean.lower() or "fidd" in text_clean.lower():
        print(f"{text_clean} -> {href}")
