import os
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from utils import text_cleaner

# === Load API Keys ===
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

SEARCH_URL = "https://api.tavily.com/search"

class CompetitorList(BaseModel):
    competitors: List[str]

function_def = {
    "name": "extract_competitors",
    "description": "Extract a list of 5 competing product names from a paragraph.",
    "parameters": {
        "type": "object",
        "properties": {
            "competitors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 5 product names"
            }
        },
        "required": ["competitors"]
    }
}

def get_competitor_list(product: str) -> List[str]:
    payload = {
        "query": f"what are 5 competitors for {product}",
        "search_depth": "basic",
        "include_answer": "basic",
        "max_results": 10
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(SEARCH_URL, json=payload, headers=headers)
    response.raise_for_status()
    raw_data = response.json()
    tavily_answer = raw_data.get("answer", "")

    prompt = f"""Given the following answer from a search engine about the product '{product}':\n\n\"{tavily_answer}\"\n\n
Extract only 5 product names that are competitors. Return them strictly in a list of strings.
"""

    chat_response = open_ai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        functions=[function_def],
        function_call="auto"
    )

    function_args = chat_response.choices[0].message.function_call.arguments

    parsed_args = CompetitorList.model_validate_json(function_args)
    competitor_list = parsed_args.competitors

    df = pd.DataFrame([{"title": item["title"], "url": item["url"]} for item in raw_data["results"]])
    df["title"] = df["title"].apply(text_cleaner)

    os.makedirs("pickle", exist_ok=True)
    df.to_pickle("pickle/competitor_list.pkl")
    print("Tavily results saved to pickle/competitor_list.pkl")

    return competitor_list

if __name__ == "__main__":
    product = "Secretlab Titan Evo 2022"
    competitors = get_competitor_list(product)
    print("Competitor List:", competitors)
