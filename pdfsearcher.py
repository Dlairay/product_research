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

# === Make language detection deterministic ===
DetectorFactory.seed = 0

@tool
def retrieve_and_process_manual(product_name: str) -> Optional[str]:
    """Pipeline tool that downloads the best user manual PDF for a product, extracts text, and saves it to a .txt file."""
    # === Search Tavily ===
    TAVILY_KEY = os.getenv("TAVILY_API_KEY")
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"{product_name} user manual pdf",
        "search_depth": "advanced",
        "max_results": 10,
        "include_domains": [
            "manualslib.com",
            "manua.ls",
            "manualzz.com",
        ]
    }

    response = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
    data = response.json()
    results = data.get("results", [])
    sorted_results = sorted(results, key=lambda r: (
        'technical-specification' not in r['url'].lower(),
        not any(keyword in r['url'].lower() for keyword in ["manual", "setup", "guide", "instruction"]),
        r['url']
    ))

    # === Extract direct PDF link ===
    for result in sorted_results:
        url = result['url']
        if url.lower().endswith(".pdf"):
            pdf_url = url
            break
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a['href']
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(url, href)
                    break
            else:
                continue
            break
        except:
            continue
    else:
        return "❌ No valid PDF URL found."

    # === Download PDF ===
    try:
        pdf_data = requests.get(pdf_url, timeout=10)
        if "application/pdf" not in pdf_data.headers.get("Content-Type", ""):
            return "❌ URL did not point to a real PDF."
        os.makedirs("others", exist_ok=True)
        filename = f"others/{product_name.replace(' ', '_')}_manual.pdf"
        with open(filename, "wb") as f:
            f.write(pdf_data.content)
    except:
        return "❌ Failed to download the PDF."

    # === Extract Text ===
    doc = fitz.open(filename)
    raw_text = "".join(page.get_text("text") or page.get_text("blocks") or "" for page in doc)
    doc.close()

    ascii_text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
    sentences = re.split(r'[.!?]\s+', ascii_text)
    english_sentences = []
    for sentence in sentences:
        if sentence.strip():
            try:
                if detect(sentence) == 'en':
                    english_sentences.append(sentence.strip())
            except:
                pass
    final_text = " ".join(english_sentences)

    # === Save to TXT ===
    if final_text.strip():
        txt_path = filename.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        return f"✅ Saved manual text to {txt_path}"
    else:
        return "⚠️ PDF was downloaded but contained no readable English text."


# === Agent wiring ===
tools = [retrieve_and_process_manual]
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def retrieve_manual_with_agent(product_name):
    prompt = f"Download and process the manual for '{product_name}', saving it to a .txt file."
    result = agent.run(prompt)
    print(result)
    return result


if __name__ == "__main__":
    product_name = "Secretlab Titan Evo Lite"
    retrieve_manual_with_agent(product_name)
