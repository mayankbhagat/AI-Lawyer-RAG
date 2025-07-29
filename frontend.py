import streamlit as st
import time
import os # Import os for path handling
import shutil # For deleting directories

# Import components from your rag_pipeline and vector_database
from rag_pipeline import answer_query, retrieve_docs, llm_model
# Import relevant functions, but not faiss_db directly here
from vector_database import load_pdf, create_chunks, get_embeddings, FAISS_DB_PATH
from langchain_community.vectorstores import FAISS # Import FAISS for its load/from_documents methods

# Ensure the pdfs_directory exists
pdfs_directory = 'pdfs/'
os.makedirs(pdfs_directory, exist_ok=True)

# Use session_state to store the FAISS database and the currently loaded PDF filename
if 'faiss_db' not in st.session_state:
    st.session_state.faiss_db = None
if 'current_pdf_filename' not in st.session_state:
    st.session_state.current_pdf_filename = None

st.title("AI Lawyer: Document Q&A")

# Clear cached database function (for new uploads)
def clear_database_cache():
    if os.path.exists(FAISS_DB_PATH):
        shutil.rmtree(FAISS_DB_PATH)
        st.session_state.faiss_db = None
        st.session_state.current_pdf_filename = None
        st.rerun() # Rerun to clear the state and UI

uploaded_file = st.file_uploader("Upload a new PDF to analyze",
                                 type="pdf",
                                 accept_multiple_files=False,
                                 on_change=clear_database_cache) # Clear cache when new file is uploaded

# Logic to process uploaded PDF and build/load FAISS DB
if uploaded_file is not None:
    # Check if this PDF is already loaded and processed
    if st.session_state.current_pdf_filename != uploaded_file.name or st.session_state.faiss_db is None:
        st.info(f"Processing '{uploaded_file.name}' for the first time or as a new document. Please wait...")

        # Save the uploaded file temporarily
        temp_pdf_path = os.path.join(pdfs_directory, uploaded_file.name)
        with open(temp_pdf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load, chunk, and embed the new PDF
            documents = load_pdf(temp_pdf_path)
            text_chunks = create_chunks(documents)
            embeddings_model = get_embeddings()

            # Create new FAISS DB for this PDF
            st.session_state.faiss_db = FAISS.from_documents(text_chunks, embeddings_model)
            st.session_state.faiss_db.save_local(FAISS_DB_PATH) # Optionally save for future runs
            st.session_state.current_pdf_filename = uploaded_file.name
            st.success(f"Successfully processed '{uploaded_file.name}'!")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.session_state.faiss_db = None
            st.session_state.current_pdf_filename = None
            # Clean up temp file in case of error
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    else:
        # If the same PDF is uploaded again, and DB exists, just load it
        if st.session_state.faiss_db is None and os.path.exists(FAISS_DB_PATH):
            st.info(f"Loading cached FAISS DB for '{uploaded_file.name}'...")
            st.session_state.faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
            st.session_state.current_pdf_filename = uploaded_file.name
            st.success("Loaded cached database!")
        elif st.session_state.faiss_db:
            st.info(f"'{uploaded_file.name}' is already loaded. Ready for questions.")


user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if st.session_state.faiss_db is not None: # Ensure DB is loaded before querying
        st.chat_message("user").write(user_query)

        start_time = time.time()

        # Pass the loaded faiss_db from session_state to retrieve_docs
        retrieved_docs = retrieve_docs(user_query, db_to_use=st.session_state.faiss_db)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        end_time = time.time()
        duration = end_time - start_time

        st.chat_message("AI Lawyer").write(response)
        st.info(f"Response generated in {duration:.2f} seconds.")

    else:
        st.error("Please upload a PDF and wait for it to be processed before asking questions.")