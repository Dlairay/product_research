from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Tuple, Optional
import os

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
        metadatas = [{"source": stringified_sources, "product": product, "type": "manual"}] * len(text_chunks) # added product and type to metadata
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



def build_label_contexts(
    product: str,
    label_data: Dict,
    db: Chroma,
    top_n: int = 5
) -> List[Dict]:
    """
    Builds context for each of the top-n complaint labels from the manual in ChromaDB.

    Args:
        product: The name of the product.
        label_data: A dictionary containing label information, including 'label_counts'.
        db: The Chroma database object.
        top_n: The number of top labels to build context for.

    Returns:
        A list of dictionaries, where each dictionary contains the label, count, and manual snippet.
        Returns an empty list on error, and logs the error.
    """
    label_counts = label_data["label_counts"]
    top_labels: List[Tuple[str, int]] = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    label_contexts: List[Dict] = []
    for label, count in top_labels:
        try:
            results = db.similarity_search(
                label,
                k=1,
                filter={"$and": [{"product": {"$eq": product}}, {"type": {"$eq": "manual"}}]}
            )
            manual_snippet: str = results[0].page_content if results else "No relevant manual section found."

            label_contexts.append({
                "product": product,
                "label": label,
                "count": count,
                "manual_snippet": manual_snippet,
            })
        except Exception as e:
            print(f"❌ Error building context for label '{label}': {e}")
            return []
    return label_contexts



def save_label_contexts(
    product: str,
    contexts: List[Dict],
    output_dir: str = "generated_contexts"
) -> None:
    """
    Saves the label contexts to a JSON file.

    Args:
        product: The name of the product.
        contexts: A list of label context dictionaries.
        output_dir: The directory to save the JSON file.

    Raises:
        OSError: If there's an issue creating the directory.
        IOError: If there's an issue writing to the file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path: str = f"{output_dir}/{product.replace(' ', '_')}_contexts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(contexts, f, indent=2)
        print(f"✅ Saved label contexts to: {out_path}")
    except OSError as e:
        print(f"❌ Error saving label contexts for '{product}': {e}")
        raise  # Re-raise to let the caller handle it
    except IOError as e:
        print(f"❌ IO Error saving label contexts for '{product}': {e}")
        raise


def load_label_contexts(
    product: str,
    output_dir: str = "generated_contexts"
) -> Optional[List[Dict]]:
    """
    Loads the label contexts from a JSON file.

    Args:
        product: The name of the product.
        output_dir: The directory where the JSON file is located.

    Returns:
        A list of label context dictionaries, or None if the file does not exist or an error occurs.
    """
    path: str = f"{output_dir}/{product.replace(' ', '_')}_contexts.json"
    try:
        if not os.path.exists(path):
            print(f"❌ Label context file not found for '{product}' at {path}")
            return None  # Explicitly return None for file not found
        with open(path, "r", encoding="utf-8") as f:
            import json
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading label contexts for '{product}': {e}")
        return None



def build_llm_prompt_from_contexts(contexts: List[Dict]) -> str:
    """
    Builds an LLM prompt from the label contexts.

    Args:
        contexts: A list of label context dictionaries.

    Returns:
        A formatted prompt string, or an empty string if contexts is empty.
    """
    if not contexts:
        return ""

    prompt: str = ""
    for item in contexts:
        prompt += (
            f"✅ {item['count']} users complained about '{item['label']}'.\n"
            f"📖 Manual says:\n\"{item['manual_snippet']}\"\n\n"
        )
    return prompt
