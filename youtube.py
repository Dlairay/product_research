import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


def youtube_data_collection(search_terms, max_result=5):
    """returns a dictionary of youtube data in the form of
    youtube_data = {"videoId": {"transcript": "transcript", 
                                "comments": ["comment1", "comment2"], 
                                "video_title": "title",
                                "comment_count": comment_count,}}
    """
    load_dotenv()
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

    items = search_response.get('items', [])
    if not items:
        print("No videos found for the given search term.")
        return youtube_data

    for item in items:
        videoId = item['id']['videoId']
        comment_data = []

        try:
            video_response = youtube.videos().list(
                part="snippet,statistics",
                id=videoId
            ).execute()
            title = video_response['items'][0]['snippet']['title']
        except (KeyError, IndexError) as e:
            print(f"Failed to fetch video details for {videoId}: {e}")
            continue

        try:
            comment_count = int(video_response['items'][0]['statistics']['commentCount'])
        except (KeyError, ValueError) as e:
            print(f"Comment count missing or invalid for {videoId}: {e}")
            comment_count = 0

        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=videoId,
                maxResults=100,
                order="relevance"
            )
            comment_response = request.execute()
            for item in comment_response.get('items', []):
                try:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comment_data.append(comment)
                except KeyError as e:
                    print(f"Skipping a malformed comment entry: {e}")
        except Exception as e:
            print(f"Failed to fetch comments for video {videoId}: {e}")

        transcript = get_transcript_from_youtube(videoId)
        video_data = {
            "transcript": transcript,
            "comments": comment_data,
            "video_title": title,
            "comment_count": comment_count
        }
        youtube_data[videoId] = video_data

    return youtube_data


def get_transcript_from_youtube(video_id):
    try:

        transcript_chunks = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([chunk['text'] for chunk in transcript_chunks])
    except (NoTranscriptFound, TranscriptsDisabled):

        return "no english transcript could be found"


# # Main execution
# search_terms = "lego batmobile 2025"
# youtube_data = youtube_data_collection(search_terms)
# print(youtube_data)

# # Save data to pickle file
# import pickle
# with open(f'{search_terms}_youtube_data.pkl', 'wb') as f:
#     pickle.dump(youtube_data, f)
