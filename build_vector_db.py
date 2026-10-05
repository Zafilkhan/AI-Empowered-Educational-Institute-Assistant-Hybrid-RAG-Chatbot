import json
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------
# LOAD CHUNKS
# --------------------------------

with open(
    "processed/all_chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


if len(chunks) == 0:

    print(
        "No chunks found. "
        "Run extraction and chunking first."
    )

    exit()


print(
    f"Loaded {len(chunks)} chunks."
)


# --------------------------------
# CONVERT CHUNKS TO LANGCHAIN DOCUMENTS
# --------------------------------

documents = []

for chunk in chunks:

    document = Document(
        page_content=chunk["text"],
        metadata={
            "source": chunk["source"],
            "url": chunk.get("url", ""),
            "page": chunk["page"]
        }
    )

    documents.append(document)


print(
    f"Created {len(documents)} LangChain documents."
)


# --------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------

print(
    "Loading embedding model..."
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# --------------------------------
# CREATE FAISS VECTOR DATABASE
# --------------------------------

print(
    "Creating FAISS vector database..."
)

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


print(
    "FAISS vector database created."
)


# --------------------------------
# CREATE VECTOR_DB FOLDER
# --------------------------------

os.makedirs(
    "vector_db",
    exist_ok=True
)


# --------------------------------
# SAVE FAISS DATABASE
# --------------------------------

vectorstore.save_local(
    "vector_db/langchain_faiss"
)


print(
    "Vector database saved successfully!"
)

print(
    "Location: vector_db/langchain_faiss"
)