import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def embed_manual_to_chromadb(
    product: str,
    label_data: dict,
    manual_text: str,
    sources: list,
    chroma_dir: str = "./chromadb"
):
    """
    Embed user manual into ChromaDB with metadata: product name, complaint labels (as string), and sources (as string).
    """
    # Prepare metadata (flatten lists into strings)
    metadata = {
        "product": product,
        "labels": ", ".join(label_data["label_counts"].keys()),
        "sources": ", ".join(sources),
        "type": "manual"
    }

    # Split manual into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(manual_text)

    # Create documents with flattened metadata
    docs = [
        Document(page_content=chunk, metadata=metadata)
        for chunk in chunks
    ]

    # Embed and store in Chroma
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(
        docs,
        embedding=embedding_function,
        persist_directory=chroma_dir,
        collection_name="product_manuals"
    )
    db.persist()
    print(f"✅ Embedded {len(docs)} chunks of '{product}' manual into ChromaDB.")


def build_label_contexts(product: str, label_data: dict, db, top_n=5):
    label_counts = label_data["label_counts"]
    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    label_contexts = []
    for label, count in top_labels:
        results = db.similarity_search(label, k=1, filter={"product": product, "type": "manual"})
        manual_snippet = results[0].page_content if results else "No relevant manual section found."

        label_contexts.append({
            "product": product,
            "label": label,
            "count": count,
            "manual_snippet": manual_snippet
        })

    return label_contexts


def save_label_contexts(product: str, contexts: list):
    os.makedirs("generated_contexts", exist_ok=True)
    out_path = f"generated_contexts/{product.replace(' ', '_')}_contexts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(contexts, f, indent=2)
    print(f"✅ Saved label contexts to: {out_path}")


def load_label_contexts(product: str):
    path = f"generated_contexts/{product.replace(' ', '_')}_contexts.json"
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def build_llm_prompt_from_contexts(contexts: list):
    prompt = ""
    for item in contexts:
        prompt += (
            f"✅ {item['count']} users complained about '{item['label']}'.\n"
            f"📖 Manual says:\n\"{item['manual_snippet']}\"\n\n"
        )
    return prompt