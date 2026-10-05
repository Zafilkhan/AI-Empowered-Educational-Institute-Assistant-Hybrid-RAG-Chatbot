import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# LOAD LANGCHAIN FAISS DATABASE
# ============================================================

print("Loading vector database...")

vectorstore = FAISS.load_local(
    "vector_db/langchain_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

all_documents = list(
    vectorstore.docstore._dict.values()
)

print(f"Loaded {len(all_documents)} documents.")


# ============================================================
# BM25 INDEX
# ============================================================

print("Building BM25 index...")

tokenized_corpus = [
    document.page_content.lower().split()
    for document in all_documents
]

bm25 = BM25Okapi(tokenized_corpus)


# ============================================================
# LOAD MISTRAL
# ============================================================

print("Loading Mistral...")

llm = ChatOllama(
    model="mistral",
    temperature=0
)


print("\nAI Assistant Ready!\n")


# ============================================================
# HELPER FUNCTION
# ============================================================

def make_clickable(path):

    abs_path = os.path.abspath(path)

    return "file:///" + abs_path.replace("\\", "/")


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query):

    query = query.lower()

    expansions = {
        "diwali": "diwali holiday academic calendar",
        "holi": "holi holiday academic calendar",
        "eid": "eid holiday academic calendar",
        "christmas": "christmas holiday academic calendar",

        "calendar": "academic calendar semester holiday examination",

        "syllabus": "syllabus curriculum course subjects units topics",

        "attendance": "attendance requirement minimum percentage policy",

        "exam": "examination exam schedule date end semester mid semester",

        "registration": "registration course registration semester registration"
    }

    expanded_query = query

    for keyword, expansion in expansions.items():

        if keyword in query:

            expanded_query += " " + expansion

    return expanded_query


# ============================================================
# DOCUMENT KEY
# ============================================================

def document_key(document):

    metadata = document.metadata

    return (
        document.page_content,
        metadata.get("source", ""),
        metadata.get("page", 0)
    )


# ============================================================
# RRF - RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    vector_documents,
    keyword_indices,
    k=60
):

    scores = {}
    documents = {}

    # --------------------------------
    # VECTOR RESULTS
    # --------------------------------

    for rank, document in enumerate(vector_documents):

        key = document_key(document)

        documents[key] = document

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank + 1)
        )

    # --------------------------------
    # BM25 RESULTS
    # --------------------------------

    for rank, index in enumerate(keyword_indices):

        document = all_documents[index]

        key = document_key(document)

        documents[key] = document

        scores[key] = scores.get(key, 0) + (
            1 / (k + rank + 1)
        )

    # --------------------------------
    # SORT
    # --------------------------------

    ranked = sorted(
        documents.items(),
        key=lambda x: scores[x[0]],
        reverse=True
    )

    return [
        documents[key]
        for key, score in ranked
    ]


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are an AI assistant for NIT Kurukshetra.

Answer the user's question using ONLY the information
provided in the context.

STRICT RULES:

1. Answer exactly what the user asked.

2. Give a precise and factual answer.

3. Keep the answer concise.

4. Do NOT use your own knowledge.

5. Do NOT guess or assume anything.

6. Do NOT invent information.

7. Do NOT combine unrelated information.

8. Use the exact names, dates, numbers,
   percentages and terminology from the context.

9. If the user asks for a date,
   provide the exact date from the context.

10. If the user asks for a name,
    provide the exact name from the context.

11. If the user asks for a number or percentage,
    provide the exact value from the context.

12. If the user asks for a list,
    provide only the relevant items.

13. If the context does not clearly contain
    the answer, say exactly:

"I could not find this information in the provided documents."

14. Never answer from general knowledge.

15. Never mention information that is not supported
    by the provided context.

16. Prefer a short direct answer over a long explanation.
"""
        ),

        (
            "human",
            """
CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""
        )
    ]
)


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    question = input("\nAsk a question: ").strip()

    # --------------------------------
    # EXIT
    # --------------------------------

    if question.lower() in ["exit", "quit"]:

        print("Goodbye!")

        break


    # --------------------------------
    # EMPTY QUESTION
    # --------------------------------

    if not question:

        print("Please enter a question.")

        continue


    # --------------------------------
    # GREETING
    # --------------------------------

    if question.lower() in ["hi", "hello", "hii"]:

        print(
            "Hello! Ask me about syllabus, "
            "calendar, exams or college documents."
        )

        continue


    # ========================================================
    # QUERY EXPANSION
    # ========================================================

    expanded_question = expand_query(question)

    print("\nExpanded Query:")
    print(expanded_question)


    # ========================================================
    # DOWNLOAD INTENT
    # ========================================================

    download_keywords = [
        "download",
        "link",
        "pdf",
        "open"
    ]

    download_intent = any(
        word in question.lower()
        for word in download_keywords
    )


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    print("\nSearching vector database...")

    vector_documents = vectorstore.similarity_search(
        expanded_question,
        k=15
    )


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    print("Searching BM25...")

    tokenized_query = expanded_question.lower().split()

    keyword_scores = bm25.get_scores(
        tokenized_query
    )

    keyword_results = sorted(
        range(len(keyword_scores)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )[:15]


    # ========================================================
    # HYBRID RETRIEVAL USING RRF
    # ========================================================

    candidates = reciprocal_rank_fusion(
        vector_documents,
        keyword_results
    )


    # Keep only strongest candidates for reranking

    candidates = candidates[:20]


    print(
        f"Hybrid retrieval candidates: {len(candidates)}"
    )


    # ========================================================
    # CROSS ENCODER RERANKING
    # ========================================================

    print("Reranking documents...")

    pairs = [
        [
            question,
            document.page_content
        ]
        for document in candidates
    ]

    scores = reranker.predict(pairs)


    reranked_documents = []

    for document, score in zip(
        candidates,
        scores
    ):

        reranked_documents.append(
            (
                document,
                float(score)
            )
        )


    reranked_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # CREATE FINAL CONTEXT
    # ========================================================

    context_parts = []

    sources = []


    print(
        "\n========== TOP RETRIEVED DOCUMENTS ==========\n"
    )


    # Only top 3 final documents

    for rank, (document, score) in enumerate(
        reranked_documents[:3],
        start=1
    ):

        text = document.page_content

        metadata = document.metadata

        source = metadata.get(
            "source",
            ""
        )

        page = metadata.get(
            "page",
            1
        )

        url = metadata.get(
            "url",
            ""
        )


        # Ignore extremely small chunks

        if len(text.strip()) < 50:

            continue


        # --------------------------------
        # DISPLAY RETRIEVED DOCUMENT
        # --------------------------------

        print(f"Rank: {rank}")
        print("Source:", source)
        print("Page:", page)
        print("Reranker Score:", score)

        print(
            "Preview:",
            text[:500]
        )

        print(
            "----------------------------------------"
        )


        # --------------------------------
        # CONTEXT
        # --------------------------------

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{text}
"""
        )


        # --------------------------------
        # SOURCE INFO
        # --------------------------------

        sources.append(
            {
                "source": source,
                "page": page,
                "url": url
            }
        )


    # ========================================================
    # FINAL CONTEXT
    # ========================================================

    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # NO CONTEXT
    # ========================================================

    if not context.strip():

        print(
            "\nAI Answer:\n"
            "I could not find this information "
            "in the provided documents."
        )

        continue


    # ========================================================
    # CREATE PROMPT
    # ========================================================

    messages = prompt.format_messages(
        context=context,
        question=question
    )


    # ========================================================
    # CALL MISTRAL
    # ========================================================

    print("\nGenerating answer...")

    try:

        response = llm.invoke(
            messages
        )

        result = response.content.strip()


    except Exception as e:

        print(
            "LLM Error:",
            e
        )

        continue


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print("\n========== AI ANSWER ==========\n")

    print(result)


    # ========================================================
    # SHOW SOURCES
    # ========================================================

    print("\n========== SOURCES ==========\n")

    for source in sources:

        print(
            "Document:",
            source["source"]
        )

        print(
            "Page:",
            source["page"]
        )

        if download_intent:

            print(
                "Download:",
                make_clickable(
                    source["url"]
                )
            )

        print(
            "---------------------"
        )