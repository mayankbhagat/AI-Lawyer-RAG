<h1>🤖 AI-Lawyer-RAG: Your Personal Legal AI Assistant ⚖</h1>

<p>
    <a href="https://legallaw-ai.streamlit.app/">
        <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
    </a>
</p>

<p>Welcome to <strong>AI-Lawyer-RAG</strong> – an innovative Retrieval-Augmented Generation (RAG) system designed to provide accurate and context-aware answers to your legal queries, leveraging the power of advanced Large Language Models and efficient document retrieval. This project aims to make legal information more accessible and understandable, transforming complex legal texts into actionable insights.</p>

<hr>

<h2>✨ Features</h2>
<ul>
    [cite_start]<li><strong>📄 Document Upload:</strong> Easily upload PDF documents (e.g., legal texts, declarations like the UDHR [cite: 2]) to serve as your knowledge base.</li>
    <li><strong>🧠 Intelligent RAG Pipeline:</strong> Our core RAG architecture ensures that responses are <strong>grounded</strong> in the provided documents, minimizing hallucinations and enhancing factual accuracy.</li>
    <li><strong>⚡ High-Powered LLM Integration:</strong> Powered by <strong>Google's Gemini API (<code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">gemini-2.5-pro</code> for generation)</strong>, providing cutting-edge conversational AI capabilities.</li>
    <li><strong>🔍 Advanced Embeddings:</strong> Utilizes <strong>Google's <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">embedding-001</code> model</strong> for creating high-dimensional vector representations, ensuring precise semantic search and document retrieval via FAISS.</li>
    <li><strong>⏱ Performance Timer:</strong> See the real-time response generation speed directly within the Streamlit interface.</li>
    <li><strong>📊 Basic Accuracy Metrics:</strong> Includes a robust <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">evaluate_rag.py</code> script to quantitatively assess context retrieval accuracy using keyword matching and a "golden dataset," demonstrating a commitment to verifiable performance.</li>
    <li><strong>☁ Cloud-Native Deployment:</strong> Seamlessly deployed on Streamlit Cloud, ensuring accessibility and scalability.</li>
</ul>

<hr>

<h2>💡 How It Works (The Technical Deep Dive)</h2>
<ol>
    <li><strong>PDF Ingestion:</strong> Users upload PDF documents (e.g., <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">UDHR.pdf</code>), which are then processed and loaded.</li>
    <li><strong>Text Chunking:</strong> The loaded documents are segmented into smaller, manageable <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">text_chunks</code> using <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">RecursiveCharacterTextSplitter</code>, optimizing for both context window limits and retrieval granularity.</li>
    <li><strong>Vector Embedding:</strong> Each <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">text_chunk</code> is transformed into a dense vector embedding using <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">GoogleGenerativeAIEmbeddings(model="embedding-001")</code>. These embeddings capture the semantic meaning of the text.</li>
    <li><strong>Vector Database (FAISS):</strong> The embeddings are then indexed and stored in a local <a href="https://faiss.ai/">FAISS</a> vector database. A caching mechanism is implemented to prevent redundant embedding generation on subsequent runs.</li>
    <li><strong>Query Embedding:</strong> When a user inputs a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">user_query</code>, it is also converted into a vector embedding using the same <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">embedding-001</code> model.</li>
    <li><strong>Similarity Search:</strong> The query embedding is used to perform a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">similarity_search</code> within the FAISS index, identifying the most semantically relevant <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">retrieved_docs</code> from the uploaded PDF.</li>
    <li><strong>Prompt Engineering:</strong> The <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">retrieved_docs</code> content is integrated into a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">custom_prompt_template</code> along with the <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">user_query</code>, instructing the LLM to answer <em>only</em> based on the provided <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">context</code>.</li>
    <li><strong>Generative AI:</strong> The <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">ChatGoogleGenerativeAI(model="gemini-2.5-pro")</code> processes this enriched prompt, generating a concise and contextually relevant <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">response</code>.</li>
    <li><strong>Performance & Accuracy Monitoring:</strong> The system tracks <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">duration</code> for query processing and includes a separate evaluation script (<code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">evaluate_rag.py</code>) for <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">context_relevance_percentage</code>.</li>
</ol>

<hr>

<h2>🚀 Get Started</h2>
<h3>Prerequisites</h3>
<ul>
    <li>Python 3.10, 3.11, or 3.12 (Python 3.13 might have compatibility issues with some libraries)</li>
    <li>A Google Cloud Project and a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">GOOGLE_API_KEY</code> from <a href="https://aistudio.google.com/">Google AI Studio</a>.</li>
</ul>

<h3>Local Installation</h3>
<ol>
    <li><strong>Clone the repository:</strong>
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">git clone https://github.com/mayankbhagat/AI-Lawyer-RAG.git
cd AI-Lawyer-RAG</code></pre>
    </li>
    <li><strong>Create and activate a virtual environment:</strong>
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source ./.venv/bin/activate</code></pre>
    </li>
    <li><strong>Install dependencies:</strong>
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">pip install -r requirements.txt</code></pre>
    </li>
    <li><strong>Set up your Google API Key:</strong>
    Create a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">.env</code> file in the root of your project and add your Google API Key:
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">GOOGLE_API_KEY="your_actual_google_api_key_here"</code></pre>
    [cite_start]<p><strong>(Remember to add <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">.env</code> to your <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">.gitignore</code>!)</strong> [cite: 112]</p>
    </li>
    <li><strong>Prepare your Vector Database:</strong>
    The project uses <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">UDHR.pdf</code> as its default document. Run the <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">vector_database.py</code> script once to build the FAISS index:
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">python vector_database.py</code></pre>
    <p>This will create the <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">vectorstore/db_faiss</code> directory.</p>
    </li>
    <li><strong>Run the Streamlit app:</strong>
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">streamlit run frontend.py</code></pre>
    <p>Your app should open in your browser!</p>
    </li>
</ol>

<hr>

<h2>🌐 Deployment</h2>
<p>This application is proudly deployed on Streamlit Cloud and is accessible via:
<br><strong><a href="https://legallaw-ai.streamlit.app/">https://legallaw-ai.streamlit.app/</a></strong></p>

<p><strong>For deployment on Streamlit Cloud, ensure your <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">GOOGLE_API_KEY</code> is added as a secret in your Streamlit Cloud app settings.</strong></p>

<hr>

<h2>🎯 Accuracy Evaluation</h2>
<p>To ensure the reliability of the AI Lawyer, a basic quantitative evaluation framework is included:</p>
<ol>
    <li><strong>Define Test Cases:</strong> A <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">eval_questions</code> dataset is embedded in <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">evaluate_rag.py</code>, containing specific <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">question</code>s, <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">ground_truth_answer</code>s, and <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">expected_context_keywords</code>.</li>
    <li><strong>Automated Context Relevance Check:</strong> The <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">evaluate_rag.py</code> script runs each question through the RAG pipeline and checks if the <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">retrieved_docs</code> contain the <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">expected_context_keywords</code>, providing a <code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">context_relevance_percentage</code>.</li>
    <li><strong>Run Evaluation:</strong>
    <pre><code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px;">python evaluate_rag.py</code></pre>
    <p>This script will output the performance metrics directly to your terminal.</p>
    </li>
</ol>
<p>This evaluation method helps to confirm that the RAG pipeline is effectively retrieving the necessary information, which is foundational for generating accurate and grounded responses from the LLM.</p>