# AI-Powered Educational Institute Assistant

A simple **Hybrid RAG-based chatbot** developed to answer questions from educational institute documents.

For the **demo and project implementation, we used NIT Kurukshetra data** as the sample knowledge base.

## Features

- PDF, HTML and Excel document processing
- LangChain-based text chunking
- FAISS semantic search
- BM25 keyword search
- Cross-Encoder reranking
- Mistral 7B using Ollama
- Streamlit interface

## Tech Stack

**Python, LangChain, FAISS, BM25, Sentence Transformers, Cross-Encoder, Mistral 7B, Ollama, Streamlit**

## How to Run

### 1. Create environment

```bash
uv venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Run the data pipeline

```bash
python extract_data.py
python chunk_data.py
python build_vector_db.py
```

### 4. Run the chatbot

```bash
python query_system.py
```

Or run the Streamlit application:

```bash
streamlit run app.py
```

## Project Team

**Team Members:**
- Zafil Khan
- Vivek Kumar

**Supervisor:**  
Dr. Vijay Kumar Sharma  
Assistant Professor

> **Note:** NIT Kurukshetra data is used only as the demo dataset for this project.
