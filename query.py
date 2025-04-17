from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv

def summarize_feedback(product_name: str, chroma_path: str = "./chromadb") -> dict:
    """
    Summarizes feedback from YouTube data stored in ChromaDB for a given product.

    Args:
        product_name (str): The name of the product to summarize feedback for.
        chroma_path (str): Path to the ChromaDB directory.

    Returns:
        dict: A dictionary with keys "result" and "source_documents".
    """ 
    load_dotenv()

    # Initialize DB and retriever
    db = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert in summarizing user feedback from YouTube videos.

Given the following context from a transcript and user comments:

{context}

Answer the question below and format your response in **JSON** with the schema:
{{
  "product": "<product name>",
  "things_people_like": ["<thing1>", "<thing2>", "..."],
  "things_to_improve": ["<thing1>", "<thing2>", "..."]
}}

Question: {question}
"""
    )

    # Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    # Query
    response = qa_chain.invoke(f"Summarize feedback about the {product_name}.")

    return {
        "result": response["result"],
        "sources": [doc.metadata.get("source", "N/A") for doc in response["source_documents"]],
        "source_documents": [doc.metadata for doc in response["source_documents"]]
    }

# Example usage
if __name__ == "__main__":
    result = summarize_feedback("nintendo switch")
    print("✅ Answer:")
    print(result["result"])
    print(f"\n📚 Source Chunks Used: {result['source_documents']}")
