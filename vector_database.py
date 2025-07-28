import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# REMOVED: from langchain_ollama import OllamaEmbeddings
# ADDED:
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure dotenv is loaded if running locally for API key
from dotenv import load_dotenv
load_dotenv()

pdfs_directory = 'pdfs/'
# REMOVED: ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def upload_pdf(file):
    with open(pdfs_directory + file.name, 'wb') as f:
        f.write(f.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path='UDHR.pdf'
documents = load_pdf(file_path) #

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
text_chunks = create_chunks(documents)

# REMOVED: ollama_model_name ="deepseek-r1:1.5b"

def get_embeddings():
    # Using Google's dedicated embedding model "embedding-001"
    # It automatically picks up GOOGLE_API_KEY from environment variables
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    return embeddings

FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embeddings())
faiss_db.save_local(FAISS_DB_PATH)