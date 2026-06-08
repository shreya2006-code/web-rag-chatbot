import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

st.set_page_config(
    page_title="Web RAG Chatbot",
    page_icon="🌐",
    layout="wide"
)

load_dotenv()
with st.sidebar:

    st.header("⚙️ Settings")
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.write("**Model:**")
    st.write("llama-3.3-70b-versatile")

    st.markdown("---")

    st.write("**Tech Stack**")
    st.write("• LangChain")
    st.write("• ChromaDB")
    st.write("• HuggingFace Embeddings")
    st.write("• Groq")
    st.write("• Streamlit")

    st.markdown("---")

    st.info("Ask questions about any webpage using RAG.")
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌐 Web RAG Chatbot")
st.caption("Ask questions about any webpage using RAG + LangChain + Groq")

example_url = st.selectbox(
    "Choose an Example Website (Optional)",
    [
        "None",
        "Wikipedia OS",
        "Python Docs",
        "LangChain Docs",
        "OpenAI Blog"
    ]
)
url_map = {
    "Wikipedia OS": "https://en.wikipedia.org/wiki/Operating_system",
    "Python Docs": "https://docs.python.org/3/",
    "LangChain Docs": "https://python.langchain.com/docs/introduction/",
    "OpenAI Blog": "https://openai.com/news/"
}
default_url = ""

if example_url != "None":
    default_url = url_map[example_url]

url = st.text_input(
    "Enter Website URL",
    value=default_url
)

question = st.text_area(
    "Ask a Question",
    height=120
)
@st.cache_resource
def create_rag_pipeline(url):

    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
if st.button("Ask"):

    with st.spinner("Processing webpage and generating answer... 🤔"):

        retriever = create_rag_pipeline(url)
        st.success("Website loaded successfully ✅")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.

If you cannot find the answer,
say "Not found in the page."

Context:
{context}

Question:
{question}

Answer:
""")

        def format_docs(docs):
            return "\n\n".join(
                d.page_content for d in docs
            )

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

    st.session_state.messages.append(
        {"question": question, "answer": answer}
    )

    for msg in st.session_state.messages:

        with st.chat_message("user"):
            st.write(msg["question"])

        with st.chat_message("assistant"):
            st.write(msg["answer"])

chat_text = ""

for msg in st.session_state.messages:
    chat_text += f"User: {msg['question']}\n\n"
    chat_text += f"AI: {msg['answer']}\n\n"
    chat_text += "-" * 50 + "\n\n"

st.download_button(
    "⬇ Download Chat",
    chat_text,
    "chat_history.txt"
)