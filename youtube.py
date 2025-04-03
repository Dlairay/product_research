import os
import json
import pandas as pd
from googleapiclient.discovery import build
import pickle
from dotenv import load_dotenv

def create_data_directory(directory_path="youtube_data"):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise Exception(f"Failed to create directory '{directory_path}': {e}")



def save_data_to_directory(data, directory_path="youtube_data"):
    create_data_directory(directory_path)
    
    # Save comments data with video URLs instead of IDs
    comments_data_with_urls = [
        {**comment, 'video_url': data['video_pages'][data['video_ids'].index(comment['video_id'])]} for comment in data['comment_data']
    ]

    # Save using Pickle
    with open(f"{directory_path}/comments_data.pkl", "wb") as f:
        pickle.dump(comments_data_with_urls, f)


def load_data_to_pandas(directory_path="youtube_data"):
    try:
        # Load using Pickle
        with open(f"{directory_path}/comments_data.pkl", "rb") as f:
            comments_data = pickle.load(f)
        comments_df = pd.DataFrame(comments_data)

        return comments_df
    except Exception as e:
        raise Exception(f"Failed to load data from '{directory_path}': {e}")


def youtube_data_collection(search_terms, max_result=5):
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    youtube = build('youtube', 'v3', developerKey=API_KEY)

    vid_id = []
    vid_page = []
    vid_title = []
    num_comments = []
    comment_data = []

    request = youtube.search().list(
        q=search_terms,
        maxResults=max_result,
        part="id",
        type="video",
        order="viewCount",  # Get top 5 most viewed videos for reliability
        relevanceLanguage='en'
    )
    search_response = request.execute()

    for i in range(max_result):
        videoId = search_response['items'][i]['id']['videoId']
        vid_id.append(videoId)
        page = "https://www.youtube.com/watch?v=" + videoId
        vid_page.append(page)

        # Fetch video details and comments in the same loop
        request = youtube.videos().list(
            part="snippet, statistics",
            id=videoId
        )
        video_response = request.execute()

        title = video_response['items'][0]['snippet']['title']
        vid_title.append(title)
        try:
            comment_count = int(video_response['items'][0]['statistics']['commentCount'])
            num_comments.append(comment_count)
        except:
            num_comments.append(0)

        # Retrieve comments if available
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=videoId,
                maxResults=100,
                order="relevance"
            )
            comment_response = request.execute()

            for item in comment_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comment_data.append({
                    'video_id': videoId,
                    'video_title': title,
                    'comment': comment
                })
        except Exception as e:
            pass

    data = {
        "video_ids": vid_id,
        "video_pages": vid_page,
        "video_titles": vid_title,
        "num_comments": num_comments,
        "comment_data": comment_data
    }

    save_data_to_directory(data)

    return data


