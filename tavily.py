
import os 
from dotenv import load_dotenv
import requests


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def webscrape(product):
 
    url = "https://api.tavily.com/search"

    payload = {
        "query": product + " review",
        "topic": "general",
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 3,
        "time_range": None,
        "days": 3,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": []
    }
    headers = {
        "Authorization": "Bearer " + TAVILY_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    response = response.json()
    answer = response["answer"]
    results_dict = {}
    for result in response["results"]:
        results_dict[result["title"]] = {"url":result["url"], 
                                         "content":result["content"]}
    return answer, results_dict






# product = "iphone 13"
# webscrape_result = webscrape(product)
# print(webscrape_result)