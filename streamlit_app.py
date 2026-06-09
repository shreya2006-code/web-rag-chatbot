import requests
from bs4 import BeautifulSoup
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
from datetime import datetime

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
if "url_history" not in st.session_state:
    st.session_state.url_history = {}

st.title("🌐 Web RAG Chatbot")
st.caption("Ask questions about any webpage using RAG + LangChain + Groq")
recent_site = st.selectbox(
    "🕒 Recent Websites",
    ["None"] + list(st.session_state.url_history.keys())
)
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

def get_page_title(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=5
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        return soup.title.string.strip()

    except:
        return "Unknown Page"

selected_url = default_url

if recent_site != "None":
    selected_url = st.session_state.url_history[recent_site]

url = st.text_input(
    "Enter Website URL",
    value=selected_url
)

if url:
    page_title = get_page_title(url)
    st.info(f"📄 Current Page: {page_title}")

question = st.text_area(
    "Ask a Question",
    height=120
)
col1, col2 = st.columns(2)

with col1:
    ask_clicked = st.button("Ask")

with col2:
    summarize_clicked = st.button("📝 Summarize Page")
st.caption("💡 Try asking:")

col1, col2 = st.columns(2)

with col1:
    st.caption("• Summarize this page")
    st.caption("• What are the key points?")

with col2:
    st.caption("• Explain like I'm a beginner")
    st.caption("• What are the advantages?")

@st.cache_resource
def create_rag_pipeline(url):

    from bs4 import SoupStrainer
    loader = WebBaseLoader(
        url,
        bs_kwargs={
            "parse_only": SoupStrainer("p")
        }
    )
    docs = loader.load()
    full_page_content = docs[0].page_content

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

    return (
        vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
        full_page_content
    )
if ask_clicked or summarize_clicked:
    if summarize_clicked:
        question = "Summarize this page with key points and takeaways"

    if not url.startswith(("http://", "https://")):
        st.error("⚠️ Please enter a valid URL")
        st.stop()

    if question.strip() == "":
        st.warning("⚠️ Please enter a question")
        st.stop()

    try:

        with st.spinner("Processing webpage and generating answer... 🤔"):

            retriever, full_page_content = create_rag_pipeline(url)

            if page_title not in st.session_state.url_history:
                st.session_state.url_history[page_title] = url

            st.success("Website loaded successfully ✅")

            retrieved_docs = retriever.invoke(question)

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )

            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the webpage context provided.

If the user asks for a summary:

Provide:

# Overview

A concise summary.

# Key Points

- Point 1
- Point 2
- Point 3

# Important Takeaways

The most important insights from the page.

If the answer is not found in the context, reply:

"Not found in the page."

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

        if summarize_clicked:

            answer = llm.invoke(
                f"""
        Summarize the following webpage.

        Provide:

        # Overview

        A concise summary.

        # Key Points

        - Point 1
        - Point 2
        - Point 3

        # Important Takeaways

        Main insights.

        Webpage Content:

        {full_page_content[:12000]}
        """
            ).content

        else:

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

       

    except Exception:

        st.error(
            "❌ Could not load webpage. Please check the URL and try again."
        )

        st.stop()

        

    st.session_state.messages.append(
        {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
            "time": datetime.now().strftime("%I:%M %p")
        }
    )
    


    for msg in st.session_state.messages:

        with st.chat_message("user"):
            st.caption(f"🕒 {msg['time']}")
            st.write(msg["question"])

        with st.chat_message("assistant"):
            st.caption(f"🕒 {msg['time']}")
            st.markdown(msg["answer"])

            word_count = len(msg["answer"].split())

            st.caption(
                f"📊 Words: {word_count} | Sources: {len(msg['sources'])}"
            )

            with st.expander("📚 Sources Used"):

                for i, doc in enumerate(msg["sources"], 1):

                    st.markdown(f"### Chunk {i}")

                    st.write(doc.page_content[:500])

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