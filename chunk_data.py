import json
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# LOAD EXTRACTED DATA
# --------------------------------

with open(
    "processed/extracted_raw.json",
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


# --------------------------------
# CONVERT DATA TO LANGCHAIN DOCUMENTS
# --------------------------------

langchain_documents = []

for doc in documents:

    document = Document(
        page_content=doc["text"],
        metadata={
            "source": doc["source"],
            "url": doc.get("url", ""),
            "page": doc["page"]
        }
    )

    langchain_documents.append(document)


print(
    "Total documents loaded:",
    len(langchain_documents)
)


# --------------------------------
# LANGCHAIN TEXT SPLITTER
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# --------------------------------
# CREATE CHUNKS
# --------------------------------

chunks = text_splitter.split_documents(
    langchain_documents
)


print(
    "Total chunks created:",
    len(chunks)
)


# --------------------------------
# SAVE CHUNKS
# --------------------------------

os.makedirs(
    "processed",
    exist_ok=True
)


# Convert LangChain Documents
# into JSON-compatible format

chunk_data = []

for chunk in chunks:

    chunk_data.append({
        "text": chunk.page_content,
        "source": chunk.metadata["source"],
        "url": chunk.metadata.get("url", ""),
        "page": chunk.metadata["page"]
    })


with open(
    "processed/all_chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunk_data,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "Chunking completed using LangChain!"
)

print(
    "Saved to: processed/all_chunks.json"
)