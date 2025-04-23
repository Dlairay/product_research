from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the DB
db = Chroma(
    persist_directory="./chromadb",
    embedding_function=OpenAIEmbeddings()
)
from langchain_chroma import Chroma

# Load the DB
db = Chroma(
    persist_directory="./chromadb",
    embedding_function=OpenAIEmbeddings()
)

print(f"🔍 Total documents in DB: {len(db.get()['documents'])}")
