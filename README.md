# AI-Empowered Educational Institute Assistant

A **Hybrid Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions from educational institute documents. The system combines semantic search, keyword-based retrieval, and document reranking to provide relevant context before generating an answer.

For the **demo and project implementation, NIT Kurukshetra data was used as the sample knowledge base**.

## Features

- Supports **PDF, HTML, and Excel** documents
- **LangChain** for document processing and text chunking
- **FAISS** for semantic vector search
- **BM25** for keyword-based retrieval
- **Cross-Encoder Reranker** for improving retrieved context
- **Sentence Transformers (`all-mpnet-base-v2`)** for document embeddings
- **Mistral 7B + Ollama** for local answer generation
- **Streamlit** interface for interactive question answering
- Source and page metadata for retrieved documents

## RAG Pipeline

```text
PDF / HTML / Excel Documents
            ↓
     Document Extraction
            ↓
   LangChain Text Chunking
            ↓
   Sentence Transformer
    (all-mpnet-base-v2)
            ↓
    ┌───────┴────────┐
    ↓                ↓
  FAISS             BM25
Semantic Search   Keyword Search
    └───────┬────────┘
            ↓
     Hybrid Retrieval
            ↓
   Cross-Encoder Reranking
            ↓
      Relevant Context
            ↓
       Mistral 7B
        via Ollama
            ↓
       Final Answer
```

## Tech Stack

**Python · LangChain · FAISS · BM25 · Sentence Transformers · Cross-Encoder · Mistral 7B · Ollama · Streamlit · uv**

## Project Structure

```text
AI-Educational-Assistant/
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
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment using **uv**:

```bash
uv venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
uv pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and download the Mistral model:

```bash
ollama pull mistral
```

Test the model:

```bash
ollama run mistral
```

## Build the Knowledge Base

Run the following commands in order:

### Extract documents

```bash
python extract_data.py
```

### Create chunks

```bash
python chunk_data.py
```

### Build the FAISS vector database

```bash
python build_vector_db.py
```

## Run the Project

### Terminal version

```bash
python query_system.py
```

### Streamlit application

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

## Example Questions

```text
Who is the Director of the institute?

When does the academic semester start?

When are the semester examinations?

What is the attendance requirement?

What courses are offered by the institute?

When is Diwali according to the academic calendar?
```

## Project Team

### Team Members

- **Zafil Khan**
- **Vivek Kumar**

### Supervisor

**Dr. Vijay Kumar Sharma**  
Assistant Professor

## Dataset / Demo Information

The project was developed as a **team project**, and **NIT Kurukshetra documents were used as the demo knowledge base** to demonstrate the RAG pipeline, retrieval system, and question-answering workflow.

The system architecture is designed to be reusable with documents from other educational institutes as well.

## License

This project is intended for **educational and academic purposes**.

The source code can be used, modified, and extended for learning and non-commercial academic projects, subject to the licenses of the third-party libraries and models used in this project.

**Note:** The NIT Kurukshetra documents/data used for demonstration may belong to their respective owners. They are included only as a project/demo knowledge source and are **not claimed as original project content**.

## Acknowledgements

This project uses open-source technologies and models including **LangChain, FAISS, Sentence Transformers, Hugging Face models, Ollama, Mistral, and Streamlit**. Please refer to their respective licenses and terms before redistribution or commercial use.
