AI-Powered Educational Institute Assistant
A Hybrid RAG-based AI chatbot that answers questions from educational institute documents using semantic search, keyword retrieval, reranking, and a local LLM.
Features
- 📄 Supports PDF, HTML, and Excel documents
- 🔎 Hybrid retrieval using FAISS + BM25
- 🎯 Cross-Encoder reranking for relevant context
- 🧩 LangChain-based document chunking
- 🤖 Mistral 7B for answer generation
- 🧠 all-mpnet-base-v2 for embeddings
- 🖥️ Streamlit-based web interface
- 📚 Source and page metadata for retrieved documents
Tech Stack
- Python
- LangChain
- FAISS
- BM25
- Sentence Transformers
- all-mpnet-base-v2
- Cross-Encoder Reranker
- Mistral 7B
- Ollama
- Streamlit
RAG Pipeline
Documents
    ↓
PDF / HTML / Excel Extraction
    ↓
LangChain Text Chunking
    ↓
all-mpnet-base-v2 Embeddings
    ↓
FAISS Semantic Search
    +
BM25 Keyword Search
    ↓
Hybrid Retrieval
    ↓
Cross-Encoder Reranking
    ↓
Relevant Context
    ↓
Mistral 7B via Ollama
    ↓
Final Answer

Project Structure
Bookly/
│
├── data/
│   ├── pdfs/
│   ├── html/
│   └── excel/
│
├── processed/
│   ├── extracted_raw.json
│   └── all_chunks.json
│
├── vector_db/
│   └── langchain_faiss/
│
├── extract_data.py
├── chunk_data.py
├── build_vector_db.py
├── query_system.py
├── rag_chatbot.py
├── app.py
├── requirements.txt
└── README.md

Installation
Create and activate the virtual environment:
uv venv

Windows:
.venv\Scripts\activate

Install dependencies:
uv pip install -r requirements.txt

Setup Ollama
Make sure Ollama is installed and running.
Pull the Mistral model:
ollama pull mistral

You can also test it with:
ollama run mistral

Build the RAG Database
Run the files in this order:
1. Extract documents
python extract_data.py

2. Create chunks
python chunk_data.py

3. Build FAISS database
python build_vector_db.py

Run the Chatbot
For terminal-based testing:
python query_system.py

For the Streamlit application:
streamlit run app.py

Then open the URL shown by Streamlit in your browser.
Example Questions
Who is the Director of the institute?

What is the academic calendar?

When are the semester examinations?

What is the attendance requirement?

When does the semester start?

What are the subjects in a particular course?

How It Works
The system first extracts information from institute documents and divides it into meaningful chunks. Each chunk is converted into an embedding using all-mpnet-base-v2 and stored in a FAISS vector database.
When a user asks a question, the system performs both semantic and keyword-based retrieval using FAISS and BM25. Retrieved documents are then ranked using a Cross-Encoder, and the most relevant context is passed to Mistral 7B through Ollama to generate the final answer.
Future Improvements
- Conversation memory
- Better table extraction
- Improved calendar/document retrieval
- Source citation in the UI
- Query rewriting and expansion
- Evaluation using a domain-specific QA dataset
- Deployment on a cloud platform
Author
Zafil Khan
GitHub: YOUR_GITHUB_LINK
