import streamlit as st
import time

from rag_pipeline import answer_query, retrieve_docs, llm_model

uploaded_file = st.file_uploader("Upload PDF",
                                 type="pdf",
                                 accept_multiple_files=False)

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)

        # Start the timer HERE, only when the button is pressed
        start_time = time.time()

        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        end_time = time.time()
        duration = end_time - start_time

        st.chat_message("AI Lawyer").write(response)
        st.info(f"Response generated in {duration:.2f} seconds.")

    else:
        st.error("Kindly upload a valid PDF file first!")