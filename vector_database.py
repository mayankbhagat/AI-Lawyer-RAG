import os
import asyncio # Make sure asyncio is imported at the very top
from dotenv import load_dotenv # Also load dotenv early

# --- START ASYNCIO EVENT LOOP FIX ---
# This block MUST come before any imports that might trigger async operations (like langchain_google_genai)
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = None

if _loop and _loop.is_running():
    # If a loop is already running, use it.
    pass
else:
    # If no loop is running or it's closed, set a new event loop.
    asyncio.set_event_loop(asyncio.new_event_loop())
# --- END ASYNCIO EVENT LOOP FIX ---

# Now import your libraries
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # This import needs loop to be ready
from langchain_community.vectorstores import FAISS


load_dotenv() # Load your environment variables from .env

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, 'wb') as f:
        f.write(f.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path='UDHR.pdf'
documents = load_pdf(file_path)

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
text_chunks = create_chunks(documents)

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    return embeddings

FAISS_DB_PATH="vectorstore/db_faiss"

# Check if the FAISS DB already exists to avoid re-embedding on every run
if os.path.exists(FAISS_DB_PATH):
    print("Loading existing FAISS DB...")
    faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS DB...")
    faiss_db = FAISS.from_documents(text_chunks, get_embeddings())
    faiss_db.save_local(FAISS_DB_PATH)
    print("FAISS DB created and saved.")