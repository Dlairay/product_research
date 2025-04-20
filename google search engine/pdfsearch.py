import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()  # read .env for TAVILY_API_KEY
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
headers = {
    "Authorization": f"Bearer {TAVILY_KEY}",
    "Content-Type": "application/json"
}

def search_user_manual(product_name):
    payload = {
        "query": f"{product_name} user manual pdf",
        "search_depth": "basic"
    }
    response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
    data = response.json()
    for result in data.get("results", []):
        url = result.get("url", "")
        print("🔎 Checking URL:", url)
        if url.lower().endswith(".pdf"):
            return url
        pdf_link = extract_pdf_from_html(url)
        if pdf_link:
            return pdf_link
        temp_pdf = download_pdf(url, save_path="others/temp_check.pdf")
        embedded_pdf_link = extract_pdf_links_from_pdf(temp_pdf)
        if embedded_pdf_link:
            return embedded_pdf_link
    return None


def extract_pdf_from_html(page_url):
    try:
        response = requests.get(page_url, timeout=10)
        if not response.ok:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                return urljoin(page_url, href)
    except Exception:
        pass
    return None


def extract_pdf_links_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri.lower().endswith(".pdf"):
                    doc.close()
                    return uri
        doc.close()
    except Exception:
        pass
    return None


def download_pdf(url, save_path="others/manual.pdf"):
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type:
            return None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception:
        return None


def try_decode_fake_pdf_as_html(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
        if content.lstrip().startswith(b"<!DOCTYPE html") or b"<html" in content[:500].lower():
            soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n").strip()
            output_path = "others/fallback_extracted_text.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(text)
            return text
    except Exception:
        pass
    return None


def get_manual_pdf(product_name, save_path="others/manual.pdf"):
    print(f"🔍 Searching for '{product_name}' user manual...")
    pdf_url = search_user_manual(product_name)
    if not pdf_url:
        print("❌ No PDF link found.")
        return None
    print(f"📎 PDF link found: {pdf_url}")
    downloaded_pdf = download_pdf(pdf_url, save_path=save_path)
    if not downloaded_pdf:
        print("❌ Failed to download PDF.")
        return None
    print(f"✅ Manual downloaded to: {downloaded_pdf}")
    return downloaded_pdf


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


if __name__ == "__main__":
    product_name = "Secretlab Titan Evo Lite"
    # download the PDF
    pdf_path = get_manual_pdf(product_name, save_path="others/manual.pdf")
    if not pdf_path:
        raise Exception(f"No PDF found for '{product_name}'")

    # extract all text
    text = extract_text_from_pdf(pdf_path)

    # ensure the output folder exists
    os.makedirs("others", exist_ok=True)

    # write raw text into manual.txt
    output_file = "others/manual.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Extracted text saved to: {output_file}")

    import re

def extract_only_english(text: str) -> str:
    """
    Remove any non‑ASCII characters (i.e. keep only English letters,
    digits, punctuation and whitespace), then collapse runs of whitespace.
    """
    # 1) Replace anything outside the 0–127 ASCII range with a space
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # 2) Collapse multiple spaces/newlines into a single space
    return re.sub(r'\s+', ' ', cleaned).strip()


