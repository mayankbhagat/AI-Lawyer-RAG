# evaluate_rag.py
import os
from dotenv import load_dotenv
load_dotenv() # Load your GOOGLE_API_KEY for local execution

# Import components from your existing RAG pipeline
# MODIFIED: We no longer import faiss_db directly from vector_database
from rag_pipeline import retrieve_docs as _retrieve_docs_internal # Rename to avoid conflict if needed later
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# ADDED: Import from vector_database for DB loading
from vector_database import get_embeddings, FAISS_DB_PATH, load_pdf, create_chunks
from langchain_community.vectorstores import FAISS # Import FAISS for load_local

# Define the LLM and Prompt template again for the evaluation context
llm_model = ChatGoogleGenerativeAI(model="gemini-pro")

# The prompt template should be consistent with what your LLM expects
template = """
You are a helpful AI assistant.
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# --- Integrated Evaluation Dataset ---
eval_questions = [
    {
        "question": "What rights does everyone have regarding nationality?",
        "ground_truth_answer": "Everyone has the right to a nationality, and no one shall be arbitrarily deprived of their nationality nor denied the right to change it.",
        "expected_context_keywords": ["nationality", "arbitrarily deprived"]
    },
    {
        "question": "What does the declaration state about slavery?",
        "ground_truth_answer": "No one shall be held in slavery or servitude; slavery and the slave trade shall be prohibited in all their forms.",
        "expected_context_keywords": ["slavery", "prohibited"]
    },
    {
        "question": "What is the primary purpose of education according to the declaration?",
        "ground_truth_answer": "Education shall be directed to the full development of the human personality and to the strengthening of respect for human rights and fundamental freedoms. It shall promote understanding, tolerance and friendship among all nations, racial or religious groups, and shall further the activities of the United Nations for the maintenance of peace.",
        "expected_context_keywords": ["education", "human personality", "respect for human rights", "peace"]
    },
    {
        "question": "What are the fundamental human rights mentioned in the Preamble?",
        "ground_truth_answer": "The Preamble mentions recognition of inherent dignity, equal and inalienable rights, freedom of speech and belief, and freedom from fear and want.",
        "expected_context_keywords": ["dignity", "inalienable rights", "freedom of speech", "fear and want"]
    },
    {
        "question": "Regarding marriage, what rights are granted to men and women?",
        "ground_truth_answer": "Men and women of full age, without any limitation due to race, nationality or religion, have the right to marry and to found a family. They are entitled to equal rights as to marriage, during marriage and at its dissolution.",
        "expected_context_keywords": ["marriage", "equal rights", "family"]
    }
]

# --- Load or Create FAISS DB for Evaluation ---
# This ensures eval_rag.py has access to the DB, which might have been created by frontend.py
print("Loading FAISS DB for evaluation...")
if not os.path.exists(FAISS_DB_PATH):
    print(f"Warning: FAISS DB not found at {FAISS_DB_PATH}. It might not have been created by frontend.py yet.")
    # Optionally, you could add logic here to create it from UDHR.pdf if it's consistently the source.
    # For a dynamic app, assume frontend handles it, or prompt user.
    print("Attempting to create a default DB from UDHR.pdf for evaluation...")
    documents = load_pdf('UDHR.pdf') # Fallback to UDHR.pdf for evaluation if no other DB exists
    text_chunks = create_chunks(documents)
    eval_faiss_db = FAISS.from_documents(text_chunks, get_embeddings())
    eval_faiss_db.save_local(FAISS_DB_PATH) # Save it for future eval runs
    print("Default FAISS DB created from UDHR.pdf for evaluation.")
else:
    eval_faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
print("FAISS DB loaded for evaluation.")

# Modified retrieve_docs for evaluation context
def retrieve_docs_for_eval(query):
    return eval_faiss_db.similarity_search(query)


def generate_answer_for_eval(question, retrieved_docs_content):
    formatted_prompt = prompt_template.format(context=retrieved_docs_content, question=question)
    response = llm_model.invoke(formatted_prompt)
    return response.content

print("--- Running RAG Evaluation ---")
results = []
correct_context_retrievals = 0

for i, data in enumerate(eval_questions):
    question = data["question"]
    ground_truth = data["ground_truth_answer"]
    expected_keywords = data.get("expected_context_keywords", [])

    print(f"\n--- Question {i+1} ---")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")

    # 1. Retrieve documents using the eval-specific retrieve_docs
    retrieved_docs = retrieve_docs_for_eval(question) # Use the eval-specific retrieve_docs
    retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Basic check for context relevance (simple keyword presence)
    context_relevant = True
    if expected_keywords:
        for keyword in expected_keywords:
            if keyword.lower() not in retrieved_content.lower():
                context_relevant = False
                break
    print(f"  Context Retrieval Relevant (Simple Keyword Check): {context_relevant}")

    if context_relevant:
        correct_context_retrievals += 1

    # 2. Generate answer
    generated_answer = generate_answer_for_eval(question, retrieved_content)
    print(f"  Generated Answer: {generated_answer}")

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated_answer,
        "context_relevant_check": context_relevant
    })

total_questions = len(eval_questions)
context_relevance_percentage = (correct_context_retrievals / total_questions) * 100 if total_questions > 0 else 0

print("\n--- Final Evaluation Summary ---")
for r in results:
    print(f"Q: {r['question']}\nA: {r['generated_answer']}\nContext Relevant: {r['context_relevant_check']}\n---")

print(f"\nOverall Context Retrieval Accuracy: {context_relevance_percentage:.2f}% ({correct_context_retrievals}/{total_questions} questions had relevant context based on keywords).")