import os
from langchain_google_genai import ChatGoogleGenerativeAI
# REMOVED: from vector_database import faiss_db # No longer import faiss_db directly
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash)

# MODIFIED: retrieve_docs now accepts 'db_to_use'
def retrieve_docs(query, db_to_use):
    return db_to_use.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

custom_prompt_template= """
use the pieces of information provided in the context to answer user's question.
if you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context
Question:{question}
Context:{context}
Answer:
"""

def answer_query(documents,model,query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain=prompt | model
    return chain.invoke({"question":query,"context":context})