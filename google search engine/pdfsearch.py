import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os

load_dotenv()  # this will read from your .env file

# Then fetch the key like this:
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
headers = {
    "Authorization": f"Bearer {TAVILY_KEY}",
    "Content-Type": "application/json"
}


def search_user_manual(product_name):
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"{product_name} user manual pdf",
        "search_depth": "basic"
    }

    response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
    data = response.json()

    for result in data.get("results", []):
        url = result.get("url", "")
        print("🔎 Checking URL:", url)  # optional, for debug


        # Step 1: Direct PDF result
        if url.lower().endswith(".pdf"):
            return url

        # Step 2: Scrape HTML for PDF links
        pdf_link = extract_pdf_from_html(url)
        if pdf_link:
            return pdf_link

        # Step 3: Download intermediate PDF and look inside for embedded links
        temp_pdf = download_pdf(url, save_path="others/temp_check.pdf")
        embedded_pdf_link = extract_pdf_links_from_pdf(temp_pdf)
        if embedded_pdf_link:
            return embedded_pdf_link

    return None
"""if the above code doesnt work, and we want it to keep finding for links till a successful one is found, indent the above search_user_manual function and try this:
def search_user_manual(product_name):
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"{product_name} user manual pdf",
        "search_depth": "basic"
    }

    response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
    data = response.json()

    for result in data.get("results", []):
        url = result.get("url", "")
        print("🔎 Checking URL:", url)

        # Step 1: Direct PDF result
        if url.lower().endswith(".pdf"):
            print("➡️ Found direct .pdf link")
            test_pdf = download_pdf(url)
            if test_pdf:
                return url

        # Step 2: Scrape HTML for PDF links
        pdf_link = extract_pdf_from_html(url)
        if pdf_link:
            print("➡️ Found PDF in HTML:", pdf_link)
            test_pdf = download_pdf(pdf_link)
            if test_pdf:
                return pdf_link

        # Step 3: Check inside downloaded file for embedded PDF
        temp_pdf = download_pdf(url, save_path="others/temp_check.pdf")
        if temp_pdf:
            embedded_pdf_link = extract_pdf_links_from_pdf(temp_pdf)
            if embedded_pdf_link:
                print("➡️ Found embedded PDF:", embedded_pdf_link)
                test_pdf = download_pdf(embedded_pdf_link)
                if test_pdf:
                    return embedded_pdf_link

        print("❌ This result did not produce a valid PDF.")

    return None
"""

def extract_pdf_from_html(page_url):
    try:
        response = requests.get(page_url, timeout=10)
        if not response.ok:
            print(f"Failed to fetch page: {page_url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                return urljoin(page_url, href)
    except Exception as e:
        print(f"Error while scraping {page_url}: {e}")
    return None

def extract_pdf_links_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            links = page.get_links()
            for link in links:
                uri = link.get("uri", "")
                if uri and uri.lower().endswith(".pdf"):
                    return uri
        doc.close()
    except Exception as e:
        print(f"Error reading PDF annotations: {e}")
    return None

def download_pdf(url, save_path="others/manual.pdf"):
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get("Content-Type", "")

        if "application/pdf" not in content_type:
            print(f"❌ Skipped: {url} is not a real PDF (Content-Type = {content_type})")
            return None

        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except Exception as e:
        print(f"Failed to download PDF from {url}: {e}")
        return None
    
from bs4 import BeautifulSoup

def try_decode_fake_pdf_as_html(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()

        # If it's actually HTML, not a binary PDF
        if content.lstrip().startswith(b"<!DOCTYPE html") or b"<html" in content[:500].lower():
            print("⚠️ File appears to be an HTML page, not a real PDF.")

            # Decode and parse it
            soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n").strip()

            # Save what we can extract
            output_path = "others/fallback_extracted_text.txt"
            with open(output_path, "w") as out:
                out.write(text)

            print(f"✅ Extracted HTML text saved to {output_path}")
            return text
        else:
            print("✅ File appears to be a real PDF.")
            return None

    except Exception as e:
        print(f"❌ Error decoding file: {e}")
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


# Example usage
if __name__ == "__main__":
    product_name = "Secretlab Titan Evo Lite"
    pdf_url = search_user_manual(product_name)

    if not pdf_url:
        raise Exception("No PDF found.")
    else:
        print("Found PDF:", pdf_url)
        pdf_path = download_pdf(pdf_url)

