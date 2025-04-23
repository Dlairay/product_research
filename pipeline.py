import os
import pandas as pd
from youtube_tool import load_YouTube_df
from tavily_tool import webscrape
from zeroshot import run_all
from data_processing import process_data
from pdfsearcher import retrieve_manual_with_agent
from chroma_tool import embed_manual_to_chromadb, build_label_contexts, save_label_contexts
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List, Dict

load_dotenv()

def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 20) -> List[str]:
    """Splits text into smaller chunks with overlap.

    Args:
        text: The text to split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The amount of overlap between chunks.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Common separators
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_manual_to_chromadb(
    product: str,
    label_data: Dict,
    text_chunks: List[str],
    sources: List[str],
    chroma_dir: str = "./chromadb",
) -> None:
    """Embeds the manual text chunks into ChromaDB.

    Args:
        product: The name of the product.
        label_data:  Dictionary containing label data.
        text_chunks:  List of text chunks.
        sources:  List of source URLs.
        chroma_dir:  Directory to store ChromaDB.
    """
    try:
        stringified_sources = ", ".join(sources)
        metadatas = [{"source": stringified_sources}] * len(text_chunks)
        Chroma.from_texts(  # Changed to class method
            texts=text_chunks,
            metadatas=metadatas,
            embedding=OpenAIEmbeddings(),
            persist_directory=chroma_dir,
        )
        print(f"[{product}] 🔗 Embedded manual into ChromaDB.")
    except Exception as e:
        print(f"[{product}] ❌ Error embedding manual: {e}")
        raise  # Re-raise to be handled by caller


def run_pipeline(product_name: str) -> Optional[str]:
    """Runs the data processing pipeline for a given product.

    Args:
        product_name: The name of the product.

    Returns:
        An optional string indicating the status of the pipeline execution.
        Returns None on success, an error message on failure.
    """
    output_path = f"pickle/combined_{product_name}_data_labeled.pkl"
    chroma_dir = "./chromadb"
    chroma_index_exists = os.path.exists(os.path.join(chroma_dir, "index"))
    manual_path = f"others/{product_name.replace(' ', '_')}_manual.txt"  # Assuming you save the manual

    if os.path.exists(output_path) and chroma_index_exists and os.path.exists(manual_path):
        print(f"[{product_name}] Labeled data, embeddings, and manual found. Skipping pipeline.")
        return "Data, embeddings, and manual found. Skipping pipeline."

    # Step 1: Load and Label Data (Compute Intensive)
    try:
        if not os.path.exists(output_path):
            print(f"[{product_name}] ⏳ Loading and labeling data...")
            youtube_df = load_YouTube_df(product_name)
            tavily_df = webscrape(product_name)
            df = pd.concat([youtube_df, tavily_df], ignore_index=True)
            labelled_df = run_all(df=df, product=product_name)
            labelled_df.to_pickle(output_path)  # Save the labeled dataframe
            dict_to_embed = process_data(labelled_df)  # from data_processing.py
            sources = list(dict_to_embed["unique_websites"])
        else:
            print(f"[{product_name}] ✅ Found existing labeled data. Loading...")
            labelled_df = pd.read_pickle(output_path)
            dict_to_embed = process_data(labelled_df)
            sources = list(dict_to_embed["unique_websites"])
    except Exception as e:
        print(f"[{product_name}] ❌ Error in data loading/labeling: {e}")
        return f"Error in data loading/labeling: {e}"

    # Step 2: Retrieve and Save Manual (Avoid Re-running Agent)
    try:
        if not os.path.exists(manual_path):
            print(f"[{product_name}] 📚 Retrieving product manual via agent...")
            manual_text_path = retrieve_manual_with_agent(product_name)
            if manual_text_path and os.path.exists(manual_text_path):
                with open(manual_text_path, "r", encoding="utf-8") as f:
                    manual_text = f.read()
                # Save the manual to avoid re-retrieval
                os.makedirs("manuals", exist_ok=True)
                with open(manual_path, "w", encoding="utf-8") as f:
                    f.write(manual_text)
            else:
                print(f"[{product_name}] ❌ No manual text found in record. Skipping embedding.")
                return None  # Or a more specific error message
        else:
            print(f"[{product_name}] ✅ Found existing manual.")
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_text = f.read()
    except Exception as e:
        print(f"[{product_name}] ❌ Error retrieving/saving manual: {e}")
        return f"Error retrieving/saving manual: {e}"

    # Step 3: Embed Manual into Chroma (Avoid Re-embedding)
    try:
        if not chroma_index_exists:
            print(f"[{product_name}] 🔗 Embedding manual into ChromaDB...")
            manual_chunks = chunk_text(manual_text)
            embed_manual_to_chromadb(
                product=product_name,
                label_data=dict_to_embed,
                text_chunks=manual_chunks,
                sources=sources,
                chroma_dir=chroma_dir,
            )
        else:
            print(f"[{product_name}] ✅ Found existing embeddings.")
    except Exception as e:
        print(f"[{product_name}] ❌ Error embedding into ChromaDB: {e}")
        return f"Error embedding into ChromaDB: {e}"

    # Step 4: Build Context (May need to re-run if labels changed significantly)
    try:
        print(f"[{product_name}] 🧩 Generating label context blocks...")
        db = Chroma(persist_directory=chroma_dir, embedding_function=OpenAIEmbeddings())
        label_contexts = build_label_contexts(product_name, dict_to_embed, db)
        save_label_contexts(product_name, label_contexts)
    except Exception as e:
        print(f"[{product_name}] ❌ Error building/saving context: {e}")
        return f"Error building/saving context: {e}"

    print(f"[{product_name}] ✅ Finished processing.\n")
    return None  # Indicate success

if __name__ == "__main__":
    product_name = "Secretlab Titan Evo 2022 Gaming Chair"
    result = run_pipeline(product_name)
    if result:
        print(f"Pipeline failed: {result}")
    # Add more products as needed
    # run_pipeline("Another Product Name")
