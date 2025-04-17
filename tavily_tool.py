import os
import requests
import asyncio
import pickle
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SEARCH_URL = "https://api.tavily.com/search"
EXTRACT_URL = "https://api.tavily.com/extract"

# -------- CLEANING FUNCTION --------
def clean_html_noise(text: str) -> str:
    if not text:
        return ""

    keywords_to_remove = [
        "Skip to main content", "Search", "Open menu", "Close menu",
        "Sign in", "Sign out", "RSS", "Newsletter", "Trending",
        "View Profile", "Subscribe", "Login", "Hamburger Menu Toggle"
    ]
    
    for k in keywords_to_remove:
        text = text.replace(k, "")
        
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# -------- TAVILY SEARCH --------
def search_tavily(product):
    payload = {
        "query": product + " review",
        "search_depth": "basic",
        "max_results": 5
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(SEARCH_URL, json=payload, headers=headers)
    return response.json().get("results", [])

# -------- BLOCKING REQUEST WRAPPED IN ASYNC --------
async def extract_single_threaded(title, url):
    def blocking_request():
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "urls": [url],
            "extract_depth": "basic",
            "include_images": False
        }
        try:
            resp = requests.post(EXTRACT_URL, json=payload, headers=headers)
            data = resp.json().get("results", [])
            return title, {
                "url": url,
                "content": data[0].get("raw_content", "") if data else ""
            }
        except Exception as e:
            print(f"[ERROR] {url}: {e}")
            return title, {
                "url": url,
                "content": ""
            }

    return await asyncio.to_thread(blocking_request)

# -------- EXTRACT ALL --------
async def extract_all(results):
    tasks = [
        extract_single_threaded(r["title"], r["url"])
        for r in results if r.get("url")
    ]
    return dict(await asyncio.gather(*tasks))

# -------- MAIN FUNCTION --------
def webscrape(product):
    filename = f"pickle/{product}_web_data.pkl"

    if os.path.exists(filename):
        print(f"[INFO] File already exists: {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)

    os.makedirs("pickle", exist_ok=True)

    search_results = search_tavily(product)
    extracted_data = asyncio.run(extract_all(search_results))

    # --- CLEAN AND CHUNK BEFORE PANDAS ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_rows = []

    for info in extracted_data.values():
        cleaned = clean_html_noise(info["content"])
        if cleaned:
            chunks = splitter.split_text(cleaned)
            for chunk in chunks:
                chunked_rows.append({
                    "content": chunk,
                    "source": "web",
                    "url": info["url"]
                })

    # Save to pickle
    with open(filename, "wb") as f:
        pickle.dump(chunked_rows, f)

    print(f"[INFO] Saved to: {filename}")
    return chunked_rows

# -------- ENTRY POINT --------
if __name__ == "__main__":
    product = "iPhone 13"
    rows = webscrape(product)
    df = pd.DataFrame(rows)
    print(df.head())
