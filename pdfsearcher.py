import os
import requests
from typing import Optional
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import fitz  # PyMuPDF
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


# === Make language detection deterministic ===
DetectorFactory.seed = 0
load_dotenv()  # read .env for TAVILY_API_KEY
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
headers = {
    "Authorization": f"Bearer {TAVILY_KEY}",
    "Content-Type": "application/json"
}

def search_user_manual(product_name):
    payload = {
        "query":  f"{product_name} detailed schematics dimensions user manual pdf",
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

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def download_pdf(url, save_path):
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

def get_manual_pdf(product_name, save_path):
    print(f"🔍 Searching for '{product_name}' user manual...")
    pdf_url = search_user_manual(product_name)
    if not pdf_url:
        print("❌ No PDF link found.")
        return None
    print(f"📎 PDF link found: {pdf_url}")
    downloaded_pdf_path = download_pdf(pdf_url, save_path=save_path)
    if not downloaded_pdf_path:
        print("❌ Failed to download PDF.")
        return None
    print(f"✅ Manual downloaded to: {downloaded_pdf_path}")
    return downloaded_pdf_path

def extract_only_english(text: str) -> str:
    """
    Remove any non‑ASCII characters.
    """
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', cleaned).strip()
@tool
def get_and_process_manual(tool_input: str) -> str:
    """
    Searches for, downloads, extracts text, and saves the user manual of a given product.

    Args:
        tool_input: A string containing the product name, likely in the format "product_name='...'".

    Returns:
        A message indicating the success or failure and the file path(s) if successful.
    """
    output_directory = "others"
    try:
        # Extract the product name from the input string
        product_name = tool_input.split("='")[1].rstrip("'")
    except IndexError:
        return f"❌ Invalid input format: '{tool_input}'. Expected 'product_name='<product>'."

    sanitized_product_name = product_name.replace(' ', '_').replace("'", "")
    pdf_filename = f"{sanitized_product_name}.pdf"
    text_filename = f"{sanitized_product_name}.txt"
    pdf_save_path = os.path.join(output_directory, pdf_filename)
    final_save_path = os.path.join(output_directory, text_filename)

    print(f"🚀 Starting the process for '{product_name}' user manual...")
    downloaded_pdf_path = get_manual_pdf(product_name, pdf_save_path)

    if not downloaded_pdf_path:
        return f"❌ Failed to find and download the user manual for '{product_name}'."

    print("📄 PDF downloaded. Extracting text...")
    text = extract_text_from_pdf(downloaded_pdf_path)
    cleaned_text = extract_only_english(text)

    os.makedirs(output_directory, exist_ok=True)
    with open(final_save_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
        print(f"✅ Successfully processed the manual for '{product_name}'. " \
           f"PDF saved to: {downloaded_pdf_path}, Text saved to: {final_save_path}")

    return final_save_path
# === Agent wiring ===
llm = ChatOpenAI(temperature=0)
tools = [get_and_process_manual]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert agent specifically tasked with finding user manuals that contain detailed schematics and dimensions for products. Your goal is to identify and process a manual that includes technical drawings with measurements."),
    ("user", "{input}"),
])

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=prompt,
    verbose=True,
)
def extract_txt_path(agent_response):
    match = re.search(r"others/([^/]+)\.txt", agent_response)
    if match:
        return match.group(0)  # Return the entire matched path
    else:
        return None

def retrieve_manual_with_agent(product_name):
    instruction = f"Find and process the user manual for '{product_name}'. Ensure the manual contains detailed schematics and dimensions. If unable to find from product name, try dropping the specific model number. Return the path to the saved text file."
    result = agent.run(instruction)
    print("Agent's raw response:", result)  # Print the raw response for debugging
    txt_path = extract_txt_path(result)
    if txt_path:
        print("Extracted text file path:", txt_path)
        return txt_path
    else:
        print("Could not extract the text file path from the agent's response.")
        return None



if __name__ == "__main__":
    product_name = "dji mini 3 pro"
    print(retrieve_manual_with_agent(product_name))
