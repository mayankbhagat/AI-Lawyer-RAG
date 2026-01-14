import os
import asyncio
from dotenv import load_dotenv

# --- START ASYNCIO EVENT LOOP FIX ---
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = None

if _loop and _loop.is_running():
    pass
else:
    asyncio.set_event_loop(asyncio.new_event_loop())
# --- END ADDED LINES ---

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS # Still import FAISS for type hinting/utility

load_dotenv()

pdfs_directory = 'pdfs/' # Make sure this directory exists for saving uploaded PDFs temporarily

def upload_pdf(file):
    # This function is now used by frontend to save the uploaded file
    # It ensures the 'pdfs/' directory exists.
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings

FAISS_DB_PATH="vectorstore/db_faiss" # Keep this constant for consistent path
# REMOVED: Direct creation/loading of faiss_db at the module level.
# This logic is now in frontend.py.