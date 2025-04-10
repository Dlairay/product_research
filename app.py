from productscan import product_scan
from youtube import youtube_data_collection
from tavily import webscrape
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pickle
import os
from processing import filter_comments_batch


product = product_scan('img/img_2.jpeg')
pickle_file = f'pickle/{product}_youtube_data.pkl'

if os.path.exists(pickle_file):
    print(f"Loading existing YouTube data for: {product}")
    with open(pickle_file, 'rb') as f:
        youtube_data = pickle.load(f)
else:
    print(f"No data found. Collecting YouTube reviews for: {product}")
    youtube_data = youtube_data_collection(product + " review", max_result=5)
    with open(pickle_file, 'wb') as f:
        pickle.dump(youtube_data, f)

# from pprint import pprint
# pprint (youtube_data)



def count_comments(youtube_data):
    count = 0
    for key, value in youtube_data.items():
        for _ in value["comments"]:
            count += 1
    print(f"Total comments: {count}")
    return count

    


def remove_junk_comments(youtube_data):
    count_comments(youtube_data)

    for key, value in youtube_data.items():
        all_comments = value["comments"]
        relevance_flags = filter_comments_batch(all_comments)

        # Keep only relevant comments
        filtered_comments = [comment for comment, is_relevant in zip(all_comments, relevance_flags) if is_relevant]
        youtube_data[key]["comments"] = filtered_comments

    count_comments(youtube_data)

remove_junk_comments(youtube_data)





