from pdfsearch import get_manual_pdf
# Before loading the PDF
from bs4 import BeautifulSoup

def try_decode_fake_pdf_as_html(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()

        if content.lstrip().startswith(b"<!DOCTYPE html") or b"<html" in content[:500].lower():
            print("⚠️ File appears to be an HTML page, not a real PDF.")

            soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n").strip()

            output_path = "others/fallback_extracted_text.txt"
            with open(output_path, "w") as out:
                out.write(text)

            print(f"✅ Extracted HTML text saved to {output_path}")
            return text
        else:
            print("✅ File appears to be a real PDF.")
            return None

    except Exception as e:
        print(f"❌ Error decoding file: {e}")
        return None

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


pdf_path = "others/manual.pdf"
import os

# Safety check to verify PDF before loading
def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            head = f.read(1024)
            return b"%PDF" in head
    except:
        return False

pdf_path = "others/manual.pdf"

if not is_valid_pdf(pdf_path):
    print(f"❌ '{pdf_path}' is not a valid PDF file. Please check your PDF search result.")
    exit()


loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
splits = splitter.split_documents(documents)
import re

def extract_matching_chunks(splits, prompt):
    prompt_keywords = prompt.lower().split()
    matching_chunks = []

    for doc in splits:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in prompt_keywords):
            matching_chunks.append(doc.page_content.strip())

    return matching_chunks


print("Splits created:", len(splits))
#create embeddings and build the vectorbase 
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

import shutil

# Delete previous vector store (if any)
try:
    shutil.rmtree('others/persist')
except:
    pass

from langchain.vectorstores import Chroma

persist_directory = 'others/persist'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


print("Vector DB created with", vectordb._collection.count(), "documents")


#reload vector db and setting up chatbot
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Reload the vector DB with the existing persistent directory
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Initialize the language model (ensure you have access to the "gpt-4o" model)
from langchain_core.messages import SystemMessage

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 8}),
    return_source_documents=True
)

# Example prompt: Adjust this query as needed.
# After getting result from LLM
prompt = "What are the specs for this chair?"

def enforce_english(prompt):
    return f"Please answer only in English. {prompt.strip()}"

result = qa_chain.invoke({"query": enforce_english(prompt)})

# Fallback: extract matching text chunks from the PDF
manual_matches = extract_matching_chunks(splits, prompt)

if manual_matches:
    # Export results to answer.txt
    with open("others/answer.txt", "w") as f:
        f.write("🔍 Prompt:\n" + prompt + "\n\n")
        f.write("🧠 GPT Answer:\n" + result["result"] + "\n\n")
        f.write("📄 Fallback Matching Chunks:\n\n")
        for chunk in manual_matches:
            f.write(chunk + "\n\n")
    print("✅ Exported GPT + fallback to answer.txt")
else:
    print("⚠️ No matching fallback text found.")


if manual_matches:
    output_path = "others/answer.txt"
    with open(output_path, "w") as f:
        f.write("🔍 Prompt:\n")
        f.write(prompt + "\n\n")
        f.write("📄 Matching Text From PDF:\n\n")
        for chunk in manual_matches:
            f.write(chunk + "\n\n")

    print(f"✅ Exported matching content to: {output_path}")
else:
    print("❌ No matching chunks found based on prompt keywords.")


result = qa_chain.invoke({"query": prompt})
gpt_answer = result["result"].strip()

# Fallback if LLM gives no good answer
if not gpt_answer or "don't have information" in gpt_answer.lower():
    print("⚠️ GPT didn't return a useful answer, searching manually...")
    manual_matches = extract_matching_chunks(splits, prompt)
    gpt_answer = "📄 Fallback answer from matching text:\n\n" + "\n\n".join(manual_matches[:3]) if manual_matches else "⚠️ No relevant text found."

# Export both
result = qa_chain.invoke({"query": prompt})
gpt_answer = result["result"].strip()

# Fallback if LLM gives no good answer
if not gpt_answer or "don't have information" in gpt_answer.lower():
    print("⚠️ GPT didn't return a useful answer, searching manually...")
    manual_matches = extract_matching_chunks(splits, prompt)
    gpt_answer = "📄 Fallback answer from matching text:\n\n" + "\n\n".join(manual_matches[:3]) if manual_matches else "⚠️ No relevant text found."

# Export both
output_path = "others/answer.txt"
with open(output_path, "w") as f:
    f.write("🔍 Prompt:\n" + prompt + "\n\n")
    f.write("🧠 Answer:\n" + gpt_answer + "\n")


print(f"✅ Answer exported to: {output_path}")




