import os
import re
import html
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

# === Standard cleaner ===
def clean_comment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 🔸 Transcript Retrieval + Chunking 🔸
def get_transcript_chunks(video_id, chunk_size=100):
    """
    Fetch and split transcript into smaller chunks.
    Returns a list of cleaned transcript chunks.
    """
    try:
        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([chunk['text'] for chunk in transcript_chunks])
        full_text = clean_comment_text(full_text)

        # Split into smaller chunks
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    except (NoTranscriptFound, TranscriptsDisabled):
        return []
    
def get_all_comments(youtube, video_id, max_total=500):
    """
    Retrieves up to max_total top-level comments for a video with pagination.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            order="relevance",
            textFormat="plainText"
        )
        while request and len(comments) < max_total:
            response = request.execute()
            for item in response.get("items", []):
                try:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                except KeyError:
                    continue
            # Advance to next page
            request = youtube.commentThreads().list_next(request, response)
    except Exception as e:
        print(f"[WARN] Failed to fetch comments for {video_id}: {e}")
    return comments

# === Main Collection Function ===
def youtube_data_to_dataframe_rows(search_terms, max_result=5, transcript_chunk_size=100):
    """
    Returns a list of dict rows ready to be used in a pandas DataFrame.
    Each row contains: content, url, source, type.
    Includes transcript chunks and comment lines.
    """
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    rows = []

    search_response = youtube.search().list(
        q=search_terms,
        maxResults=max_result,
        part="id",
        type="video",
        order="viewCount",
        relevanceLanguage='en'
    ).execute()

    items = search_response.get('items', [])
    print(len(items), "videos found")
    if not items:
        print("No videos found for the given search term.")
        return rows

    for item in items:
        videoId = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={videoId}"

        # 🔸 Add transcript chunks 🔸
        transcript_chunks = get_transcript_chunks(videoId, chunk_size=transcript_chunk_size)
        for chunk in transcript_chunks:
            rows.append({
                "content": chunk,
                "type": "transcript",
                "source": "YouTube",
                "url": video_url
            })

        # === Add comments ===
        comment_data = get_all_comments(youtube, videoId, max_total=200)
        for comment in comment_data:
            rows.append({
                "content": clean_comment_text(comment),
                "type": "comment",
                "source": "YouTube",
                "url": video_url
            })


    return rows






def load_YouTube_df(search_terms: str, max_result: int = 5, transcript_chunk_size: int = 100, cache_dir: str = "pickle") -> pd.DataFrame:
    """
    Loads a YouTube DataFrame from cache if it exists, else builds and saves it.
    Returns the final DataFrame with columns: content, url, source, type
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Safe filename based on search term
    safe_name = search_terms.replace(" ", "_").replace("/", "_")
    filepath = os.path.join(cache_dir, f"{safe_name}_youtube_df.pkl")

    if os.path.exists(filepath):
        print(f"[INFO] Loading cached DataFrame from: {filepath}")
        return pd.read_pickle(filepath)

    print(f"[INFO] No cache found. Collecting data for: {search_terms}")
   

    rows = youtube_data_to_dataframe_rows(search_terms, max_result=max_result, transcript_chunk_size=transcript_chunk_size)
    df = pd.DataFrame(rows, columns=["content", "source", "type", "url"])
    
    df.to_pickle(filepath)
    print(f"[INFO] Saved new DataFrame to: {filepath}")

    return df


if __name__ == "__main__":
    # Example usage
    search_terms = "Secretlab Titan Evo 2022 Gaming Chair"
    youtube_df = load_YouTube_df(search_terms)

    print(youtube_df.head())
    print(youtube_df.columns)
    print(youtube_df.shape)
    print(youtube_df['type'].unique())

    # ✅ Count number of comments vs transcript rows
    print("\n[INFO] Row counts by type:")
    print(youtube_df['type'].value_counts())
