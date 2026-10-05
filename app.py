import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Empowered Educational Institute Assistant — Hybrid RAG Chatbot",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f7fa;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">llabus, exa🎓 AI-Empowered Educational Institute Assistant — Hybrid RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about college documents, syms, '
    'academic calendar and more.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_system():

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # FAISS database
    vectorstore = FAISS.load_local(
        "vector_db/langchain_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # Mistral
    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant for a college website.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context, say:

"I could not find this information in the documents."

Be concise and clear.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return retriever, llm, prompt


# --------------------------------------------------
# LOAD SYSTEM
# --------------------------------------------------

try:

    retriever, llm, prompt = load_system()

except Exception as e:

    st.error(
        "Could not load the RAG system."
    )

    st.code(str(e))

    st.stop()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about NIT Kurukshetra..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------
    # RAG
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching college documents..."
        ):

            try:

                # Retrieve documents
                documents = retriever.invoke(
                    question
                )


                # Create context
                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )


                # Create prompt
                messages = prompt.format_messages(
                    context=context,
                    question=question
                )


                # Ask Mistral
                response = llm.invoke(
                    messages
                )


                answer = response.content


                # Display answer
                st.markdown(answer)


                # Save answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


                # --------------------------------------------------
                # SOURCES
                # --------------------------------------------------

                with st.expander(
                    "📚 Sources"
                ):

                    for i, document in enumerate(
                        documents
                    ):

                        source = document.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = document.metadata.get(
                            "page",
                            "Unknown"
                        )

                        st.write(
                            f"**{i + 1}. {source}**"
                        )

                        st.write(
                            f"Page: {page}"
                        )

                        st.write(
                            document.page_content[:300]
                            + "..."
                        )

                        st.divider()


            except Exception as e:

                st.error(
                    "Error while generating answer."
                )

                st.code(
                    str(e)
                )