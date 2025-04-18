import os
import re
import html
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

# === Environment & API key ===
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# === Standard cleaner ===
def clean_comment_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Transcript Retrieval + Chunking ===
def get_transcript_chunks(video_id, chunk_size=100):
    try:
        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([c['text'] for c in transcript_chunks])
        full_text = clean_comment_text(full_text)
        return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size) if full_text[i:i+chunk_size].strip()]
    except (NoTranscriptFound, TranscriptsDisabled):
        return []

# === Comments Retrieval ===
def get_all_comments(youtube, video_id, max_total=500):
    comments = []
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        while req and len(comments) < max_total:
            res = req.execute()
            for item in res.get('items', []):
                txt = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(txt)
            req = youtube.commentThreads().list_next(req, res)
    except Exception:
        pass
    return comments

# === Main Collection Function ===
def youtube_data_to_dataframe_rows(search_terms, max_result=5, transcript_chunk_size=100):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    rows = []
    search_response = youtube.search().list(
        q=search_terms,
        maxResults=max_result,
        part='snippet'
    ).execute()

    for item in search_response.get('items', []):
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Transcript chunks
        for chunk in get_transcript_chunks(video_id, transcript_chunk_size):
            rows.append({
                'content': chunk,
                'type': 'transcript',
                'source': 'YouTube',
                'url': video_url
            })

        # Comment rows
        for comment in get_all_comments(youtube, video_id):
            clean = clean_comment_text(comment)
            if clean:
                rows.append({
                    'content': clean,
                    'type': 'comment',
                    'source': 'YouTube',
                    'url': video_url
                })
    return rows

# === Loader & DataFrame builder ===
def load_YouTube_df(search_terms: str, max_result: int = 5, transcript_chunk_size: int = 100, cache_dir: str = 'pickle') -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    safe_fn = re.sub(r"[^0-9a-zA-Z]+", "_", search_terms)
    path = os.path.join(cache_dir, f"{safe_fn}_youtube.pkl")

    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        rows = youtube_data_to_dataframe_rows(search_terms, max_result, transcript_chunk_size)
        df = pd.DataFrame(rows, columns=['content','type','source','url'])
        df.to_pickle(path)
    return df

if __name__ == '__main__':
    df = load_YouTube_df("Secretlab Titan Evo 2022 Gaming Chair")
    print(df.head())
