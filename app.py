from productscan import product_scan
from youtube_tool import youtube_data_retrieval
from tavily_tool import webscrape
import pickle
import pandas as pd

# from google.adk.agents import Agent
# from google.adk.models.lite_llm import LiteLlm
# from vectordb import youtube_to_chromadb
# from query import summarize_feedback
# from tavily_tool import search_competitor_names




# product = product_scan("img/img_3.jpeg")
product = "Secretlab Titan Evo 2022 Gaming Chair"

youtube_data = youtube_data_retrieval(product)
# ##add youtube data to central pandas dataframe
# Convert data to a DataFrame
rows = []
for video_id, video_data in youtube_data.items():
    for comment in video_data.get("comments", []):
        rows.append({
            "content": comment,
            "source": "YouTube",
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })

youtube_df = pd.DataFrame(rows)

# # Show the first few rows
# print(df.head())

tavily_data = webscrape(product)


with open("pickle/Secretlab Titan Evo 2022 Gaming Chair_web_data.pkl", "rb") as f:
    web_data = pickle.load(f)

# Convert to DataFrame
rows = []
for title, info in web_data.items():
    rows.append({
        "content": info.get("content", ""),
        "source": "Web",
        "url": info.get("url", "")
    })

web_df = pd.DataFrame(rows)
print(web_df.head())

# combined_df = pd.concat([youtube_df, web_df], ignore_index=True)


# combined_df.to_csv("combined_data.csv", index=False)