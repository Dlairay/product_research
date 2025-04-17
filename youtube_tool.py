import os
import pickle
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_openai import ChatOpenAI

load_dotenv()

# === Utility: YouTube Transcript Fetch ===
def get_transcript_from_youtube(video_id):
    """gets transcript from youtube video id"""
    try:
        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([chunk['text'] for chunk in transcript_chunks])
    except (NoTranscriptFound, TranscriptsDisabled):
        return "no english transcript could be found"


# === Tool 1: YouTube Data Collection ===
def youtube_data_collection(search_terms: str, max_result: int = 5) -> dict:
    """
    Collects YouTube video metadata, comments, and transcripts based on a search term.
    """
    API_KEY = os.getenv("GOOGLE_API_KEY")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    youtube_data = {}

    search_response = youtube.search().list(
        q=search_terms,
        maxResults=max_result,
        part="id",
        type="video",
        order="viewCount",
        relevanceLanguage='en'
    ).execute()

    for item in search_response.get('items', []):
        videoId = item['id']['videoId']
        comment_data = []
        title = "Unknown Title"
        comment_count = 0

        try:
            video_response = youtube.videos().list(
                part="snippet,statistics",
                id=videoId
            ).execute()
            title = video_response['items'][0]['snippet']['title']
            comment_count = int(video_response['items'][0]['statistics'].get('commentCount', 0))
        except Exception as e:
            print(f"Error retrieving video info for {videoId}: {e}")

        try:
            comment_response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=videoId,
                maxResults=100,
                order="relevance"
            ).execute()
            for item in comment_response.get('items', []):
                try:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comment_data.append(comment)
                except KeyError:
                    continue
        except Exception as e:
            print(f"Error retrieving comments for {videoId}: {e}")

        transcript = get_transcript_from_youtube(videoId)
        video_data = {
            "transcript": transcript,
            "comments": comment_data,
            "video_title": title,
            "comment_count": comment_count,

        }
        youtube_data[videoId] = video_data
        youtube_data["filtered_comment_flag"] = False

    return youtube_data


# === Tool 2: Comment Filtering with LLM ===
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def filter_comments_batch(comments: list[str], batch_size: int = 10) -> list[bool]:
    """
    Uses an LLM to determine if YouTube comments are relevant to the product.
    Returns a list of True/False flags.
    """
    results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        formatted = "\n".join([f"{idx+1}. {c}" for idx, c in enumerate(batch)])
        prompt = f"""
You are helping filter YouTube comments for product relevance.

Return a list of 1s and 0s matching the order of comments:
- 1 = relevant to the product
- 0 = not relevant

Examples:
Relevant: "Battery life is amazing"
Irrelevant: "I love your videos"

Comments:
{formatted}

Return format: [1, 0, 1, ...]
"""
        response = llm.invoke(prompt).content.strip()
        try:
            binary = eval(response)
            results.extend([str(val).strip() == '1' for val in binary])
        except Exception as e:
            print("Parse error:", response)
            results.extend([False] * len(batch))
    return results




# === Tool 3: End-to-End YouTube Retrieval Pipeline tool===
def youtube_data_retrieval(product_name: str) -> dict:
    import time
    start_time = time.time()
    print("loading youtube data...")
    
    """
    Full pipeline: searches YouTube, collects metadata and comments, filters out irrelevant ones.
    Saves to pickle for caching.
    """
    pickle_file = f'pickle/{product_name}_youtube_data.pkl'

    if os.path.exists(pickle_file):
        print(f"Loading existing YouTube data for: {product_name}")
        with open(pickle_file, 'rb') as f:
            youtube_data = pickle.load(f)
    else:
        print(f"No data found. Collecting YouTube reviews for: {product_name}")
        youtube_data = youtube_data_collection(product_name + " review") #👈🏻 calling the first function
        with open(pickle_file, 'wb') as f:
            pickle.dump(youtube_data, f)

    def count_comments(data):
        return sum(len(d["comments"]) for d in data.values())
    if youtube_data.get("filtered_comment_flag", False):
        print("Before filtering:", count_comments(youtube_data))

        for key, value in youtube_data.items():
            all_comments = value["comments"]
            relevance_flags = filter_comments_batch(all_comments) #👈🏻 calling the second function
            filtered_comments = [c for c, r in zip(all_comments, relevance_flags) if r]
            youtube_data[key]["comments"] = filtered_comments
        youtube_data["filtered_comment_flag"] = True
        print("After filtering:", count_comments(youtube_data))
    else:
        print("Already filtered, skipping...")
    print("time taken to load youtube data:", time.time() - start_time)
    return youtube_data



if __name__ == "__main__":
    product = "Secretlab Titan Evo 2022 Gaming Chair"
    youtube_data = youtube_data_retrieval(product)
    print(youtube_data)
    # with open('full_dump.json', 'w', encoding='utf-8') as f:
    #     json.dump(youtube_data, f, indent=2, ensure_ascii=False)

