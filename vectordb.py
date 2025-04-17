import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def youtube_to_chromadb(product: str, chroma_dir: str = "./chromadb"):
    """
    Loads YouTube data from a pickle file for a given product, splits it into transcript and comment chunks,
    embeds them using OpenAIEmbeddings, and stores in a unified ChromaDB with product metadata.
    
    Args:
        product (str): Product name used to construct the pickle file path.
        chroma_dir (str): Directory path to store persistent ChromaDB.
    """
    load_dotenv()

    pickle_path = f"pickle/{product}_youtube_data.pkl"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"❌ Pickle file not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        videos = pickle.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []

    for video_id, content in videos.items():
        # Split transcript
        if "transcript" in content:
            transcript_chunks = splitter.create_documents(
                [content["transcript"]],
                metadatas=[{
                    "source": "YouTube",
                    "product": product,
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
                    "source": f"youtube comment:https://www.youtube.com/watch?v={video_id}",
                    "product": product,
                    "video_id": video_id,
                    "video_title": content.get("video_title", ""),
                    "chunk_type": "comment",
                    "comment_index": i
                }]
            )
            docs.extend(comment_chunks)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=chroma_dir)


    print(f"✅ Done! Stored {len(docs)} chunks from '{product}' into ChromaDB at {chroma_dir}.")
