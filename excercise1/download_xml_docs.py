import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
RAW_DIR = "documents"

os.makedirs(RAW_DIR, exist_ok=True)

def list_xml_files():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    files = [
        a["href"]
        for a in soup.find_all("a")
        if a.get("href", "").endswith(".xml")
    ]
    return sorted(files)

def download_selected(start_name="debates2023-06-28d.xml"):
    all_files = list_xml_files()
    selected = [f for f in all_files if f >= start_name]
    print(f"Selected {len(selected)} XML files.")
    for name in selected:
        url = urljoin(BASE_URL, name)
        out_path = os.path.join(RAW_DIR, name)
        if os.path.exists(out_path):
            continue
        r = requests.get(url)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)

if __name__ == "__main__":
    download_selected()
    print("Download complete.")
