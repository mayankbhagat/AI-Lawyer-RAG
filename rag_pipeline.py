# REMOVED: from langchain_groq import ChatGroq
# ADDED:
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Initialize the Google Gemini LLM
# 'gemini-pro' is a powerful, general-purpose model suitable for chat and text generation.
# It automatically uses the GOOGLE_API_KEY environment variable.
llm_model = ChatGoogleGenerativeAI(model="gemini-pro")


def retrieve_docs(query):
    return faiss_db.similarity_search(query)

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