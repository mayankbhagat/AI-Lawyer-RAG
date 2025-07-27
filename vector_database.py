# vector_database.py
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# REMOVE THIS LINE: from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ADD THESE LINES for OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv() # Ensure .env is loaded for API key

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, 'wb') as f:
        f.write(file.getbuffer())

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

# REMOVE THIS LINE: ollama_model_name ="deepseek-r1:1.5b"

def get_embeddings(): # No longer needs a model name for OpenAI default
    # Use OpenAIEmbeddings instead of OllamaEmbeddings
    # It will automatically pick up OPENAI_API_KEY from environment variables
    embeddings = OpenAIEmbeddings()
    return embeddings

FAISS_DB_PATH="vectorstore/db_faiss"
# Call get_embeddings without the ollama_model_name argument
faiss_db=FAISS.from_documents(text_chunks, get_embeddings())
faiss_db.save_local(FAISS_DB_PATH)