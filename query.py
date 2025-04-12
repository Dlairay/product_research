from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


db = Chroma(persist_directory="./chromadb_youtube", embedding_function=OpenAIEmbeddings())

# Create retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Use a language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 temperature=0)

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

# Create the RetrievalQA chain with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# response = qa_chain.invoke("Summarize feedback about the Secretlab Titan Evo chair.")

if __name__ == "__main__":
    response = qa_chain.invoke("Summarize feedback about the Secretlab Titan Evo chair.")

    print("✅ Answer:")
    print(response["result"])

    print("\n📚 Source Chunks Used:")
    for doc in response["source_documents"]:
        print(doc.metadata)
        print(doc.page_content[:200], "...")  # Optional: truncate long chunks