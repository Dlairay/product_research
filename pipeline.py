import os
import pandas as pd
from youtube_tool import load_YouTube_df
from tavily_tool import webscrape
from zeroshot import run_all
from data_processing import process_data
from pdfsearcher import retrieve_manual_with_agent
from chroma_tool import  embed_manual_to_chromadb,build_label_contexts,save_label_contexts
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def run_pipeline(product_name: str):
    print(f"\n🚀 Starting pipeline for: {product_name}")

    # Step 1: Load or fetch data
    combined_path = f"pickle/combined_{product_name}_data_labeled.pkl"
    if os.path.exists(combined_path):
        print(f"[{product_name}] 📂 Loading cached data...")
        combined_df = pd.read_pickle(combined_path)
    else:
        print(f"[{product_name}] 🌐 Fetching YouTube and web data...")
        # youtube_df = load_YouTube_df(product_name, max_result=5, transcript_chunk_size=100, cache_dir='pickle')
        tavily_df = webscrape(product_name)
        combined_df = pd.concat([tavily_df], ignore_index=True)
        combined_df.to_pickle(combined_path)
        print(f"[{product_name}] ✅ Combined data saved to {combined_path}")

    # Step 2: Run zero-shot classifier
    print(f"[{product_name}] 🧠 Running zero-shot classification...")
    labelled_df, labels_list = run_all(combined_df)
    dict_to_embed = process_data(labelled_df)
    sources = list(dict_to_embed["unique_websites"])

    # Step 3: Retrieve manual
    print(f"[{product_name}] 📚 Retrieving product manual via agent...")
    manual_text_path = retrieve_manual_with_agent(product_name)

    if not manual_text_path or not os.path.exists(manual_text_path):
        print(f"[{product_name}] ❌ No manual text found. Skipping embedding.")
        return None

    with open(manual_text_path, "r", encoding="utf-8") as f:
        manual_text = f.read()

    # Step 4: Embed into Chroma
    print(f"[{product_name}] 🔗 Embedding manual into ChromaDB...")
    embed_manual_to_chromadb(
        product=product_name,
        label_data=dict_to_embed,
        manual_text=manual_text,
        sources=sources,
        chroma_dir="./chromadb"
    )

    # Step 5: Build context
    print(f"[{product_name}] 🧩 Generating label context blocks...")
    db = Chroma(persist_directory="./chromadb", embedding_function=OpenAIEmbeddings())
    label_contexts = build_label_contexts(product_name, dict_to_embed, db)
    save_label_contexts(product_name, label_contexts)

    print(f"[{product_name}] ✅ Finished processing.\n")
    return labelled_df
