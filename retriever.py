from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# Custom RAG prompt template
rag_prompt = PromptTemplate.from_template("""
You are an expert product design researcher.

Using the following manual content, suggest specific improvements to the product
that would address common user complaints.

If you cannot find relevant context in the manual, respond with "No suggestions found".

Context:
{context}

Question:
{question}
""")

def get_retrieval_qa(product: str):
    db = Chroma(
    persist_directory="./chromadb",
    collection_name="product_manuals",
    embedding_function=OpenAIEmbeddings()
)

    retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"product": product}})
    llm = ChatOpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )
    return qa_chain

def run_suggestions(product: str):
    print(f"💡 Generating suggestions for: {product}")
    qa_chain = get_retrieval_qa(product)

    question = (
        "How can this product be improved based on the top user complaints? suggest specific changes to particular components or features with sizes,numbers,material change recommendations,etc using the manual content."
    )

    response = qa_chain.invoke({"query": question})
    print(f"Retrieved documents: {response['source_documents']}")  # Print source documents
    print(f"\n🧠 Suggestions:\n{response['result']}\n")
    return response["result"]
