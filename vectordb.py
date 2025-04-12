import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# --- load OpenAI API key ---
load_dotenv()
# --- Load the pickle file need a way to automatatically retrieve this dynamically ---
with open("pickle/Secretlab Titan Evo 2022 Gaming Chair_youtube_data.pkl", "rb") as f:
    videos = pickle.load(f)  # This is the big dict

# --- Text splitter ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = []

for video_id, content in videos.items():
    # Split transcript
    if "transcript" in content:
        transcript_chunks = splitter.create_documents(
            [content["transcript"]],
            metadatas=[{
                "source": "YouTube",
                "video_id": video_id,
                "video_title": content.get("video_title", ""),
                "chunk_type": "transcript"
            }]
        )
        docs.extend(transcript_chunks)

    # Split comments
    for i, comment in enumerate(content.get("comments", [])):
        comment_chunks = splitter.create_documents(
            [comment],
            metadatas=[{
                "source": "YouTube",
                "video_id": video_id,
                "video_title": content.get("video_title", ""),
                "chunk_type": "comment",
                "comment_index": i
            }]
        )
        docs.extend(comment_chunks)

# --- Embed and store in Chroma ---
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chromadb_youtube")
db.persist()

print(f"✅ Done! Stored {len(docs)} chunks from .pkl into ChromaDB.")
