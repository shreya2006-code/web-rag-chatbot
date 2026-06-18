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
import time
import uuid

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.set_page_config(
    page_title="Web RAG Chatbot",
    page_icon="🌐",
    layout="wide"
)

load_dotenv()

# ─── Session State Init ────────────────────────────────────────────────────────

if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {
            "name": "Chat 1",
            "messages": [],
            "url_history": {},
            "current_url": None,
            "current_retriever": None,
            "current_page_content": None,
            "website_urls": [""],
        }
    }
    st.session_state.chat_order = [first_id]
    st.session_state.active_chat = first_id

if "chat_order" not in st.session_state:
    st.session_state.chat_order = list(st.session_state.chats.keys())

if "active_chat" not in st.session_state:
    st.session_state.active_chat = st.session_state.chat_order[0]

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "rename_chat_id" not in st.session_state:
    st.session_state.rename_chat_id = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_active():
    return st.session_state.chats[st.session_state.active_chat]

def new_chat():
    cid = str(uuid.uuid4())
    n = len(st.session_state.chat_order) + 1
    st.session_state.chats[cid] = {
        "name": f"Chat {n}",
        "messages": [],
        "url_history": {},
        "current_url": None,
        "current_retriever": None,
        "current_page_content": None,
        "website_urls": [""],
    }
    st.session_state.chat_order.append(cid)
    st.session_state.active_chat = cid
    st.session_state.rename_chat_id = None

def delete_chat(cid):
    if len(st.session_state.chat_order) == 1:
        # Reset instead of delete
        st.session_state.chats[cid] = {
            "name": "Chat 1",
            "messages": [],
            "url_history": {},
            "current_url": None,
            "current_retriever": None,
            "current_page_content": None,
            "website_urls": [""],
        }
        return
    st.session_state.chat_order.remove(cid)
    del st.session_state.chats[cid]
    st.session_state.active_chat = st.session_state.chat_order[-1]

def get_page_title(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.title.string.strip()
    except:
        return url

@st.cache_resource
def create_rag_pipeline(urls_tuple):
    from bs4 import SoupStrainer
    all_docs = []
    full_page_content = ""

    for single_url in urls_tuple:
        if not single_url.strip():
            continue
        try:
            loader = WebBaseLoader(
                single_url,
                bs_kwargs={"parse_only": SoupStrainer("p")}
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_url"] = single_url
            all_docs.extend(docs)
            if docs:
                full_page_content += f"\n\n[Source: {single_url}]\n" + docs[0].page_content
        except Exception as e:
            st.warning(f"⚠️ Could not load {single_url}: {e}")

    if not all_docs:
        return None, ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return (
        vectorstore.as_retriever(search_kwargs={"k": 20}),
        full_page_content
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.session_state.search_query = st.text_input(
        "🔍 Search chats",
        value=st.session_state.search_query,
        placeholder="Search by name..."
    )

    st.markdown("---")

    for cid in st.session_state.chat_order:
        chat = st.session_state.chats[cid]
        if st.session_state.search_query.lower() not in chat["name"].lower():
            continue

        is_active = cid == st.session_state.active_chat

        col_btn, col_ren, col_del = st.columns([5, 1, 1])

        with col_btn:
            label = f"{'▶ ' if is_active else ''}{chat['name']}"
            if st.button(label, key=f"sel_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.session_state.rename_chat_id = None
                st.rerun()

        with col_ren:
            if st.button("✏️", key=f"ren_{cid}", help="Rename"):
                st.session_state.rename_chat_id = cid
                st.rerun()

        with col_del:
            if st.button("🗑️", key=f"del_{cid}", help="Delete"):
                delete_chat(cid)
                st.rerun()

        # Inline rename input
        if st.session_state.rename_chat_id == cid:
            new_name = st.text_input(
                "New name",
                value=chat["name"],
                key=f"rename_input_{cid}"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Save", key=f"save_ren_{cid}"):
                    st.session_state.chats[cid]["name"] = new_name.strip() or chat["name"]
                    st.session_state.rename_chat_id = None
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key=f"cancel_ren_{cid}"):
                    st.session_state.rename_chat_id = None
                    st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        get_active()["messages"] = []
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

# ─── Main UI ──────────────────────────────────────────────────────────────────

active = get_active()

st.title(f"🌐 {active['name']}")
st.caption("Ask questions about any webpage using RAG + LangChain + Groq")

# Recent websites
recent_site = st.selectbox(
    "🕒 Recent Websites",
    ["None"] + list(active["url_history"].keys())
)

example_url = st.selectbox(
    "Choose an Example Website (Optional)",
    ["None", "Wikipedia OS", "Python Docs", "LangChain Docs", "OpenAI Blog"]
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

selected_url = default_url
if recent_site != "None":
    selected_url = active["url_history"][recent_site]

# ─── URL Inputs with Delete Buttons ───────────────────────────────────────────

# Ensure at least one URL slot
if "website_urls" not in active or not active["website_urls"]:
    active["website_urls"] = [""]

# Apply selected_url to first slot if user picked example/recent
if selected_url and active["website_urls"][0] == "":
    active["website_urls"][0] = selected_url

urls_to_delete = []
updated_urls = []

st.markdown("### 🌐 Website URLs")

for i, url_val in enumerate(active["website_urls"]):
    col_input, col_del = st.columns([11, 1])
    with col_input:
        new_val = st.text_input(
            f"Website URL {i + 1}",
            value=url_val,
            key=f"url_input_{st.session_state.active_chat}_{i}"
        )
        updated_urls.append(new_val)
    with col_del:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        if st.button("❌", key=f"del_url_{st.session_state.active_chat}_{i}", help="Remove this URL"):
            urls_to_delete.append(i)
        st.markdown("</div>", unsafe_allow_html=True)

# Apply deletions
if urls_to_delete:
    active["website_urls"] = [u for idx, u in enumerate(updated_urls) if idx not in urls_to_delete]
    if not active["website_urls"]:
        active["website_urls"] = [""]
    st.rerun()
else:
    active["website_urls"] = updated_urls

if st.button("➕ Add Website"):
    active["website_urls"].append("")
    st.rerun()

# Show current page title
url_first = active["website_urls"][0] if active["website_urls"] else ""
if url_first.strip():
    page_title = get_page_title(url_first)
    st.info(f"📄 Current Page: {page_title}")
else:
    page_title = ""

# ─── Question & Buttons ───────────────────────────────────────────────────────

question = st.text_area("Ask a Question", height=120)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("Ask")
with col2:
    summarize_clicked = st.button("📝 Summarize Page")

st.caption("💡 Try asking:")
c1, c2 = st.columns(2)
with c1:
    st.caption("• Summarize this page")
    st.caption("• What are the key points?")
with c2:
    st.caption("• Explain like I'm a beginner")
    st.caption("• What are the advantages?")

# ─── RAG Logic ────────────────────────────────────────────────────────────────

if ask_clicked or summarize_clicked:
    if summarize_clicked:
        question = "Summarize this page with key points and takeaways"

    valid_urls = [u.strip() for u in active["website_urls"] if u.strip()]

    for u in valid_urls:
        if not u.startswith(("http://", "https://")):
            st.error(f"⚠️ Invalid URL: {u}")
            st.stop()

    if not valid_urls:
        st.warning("⚠️ Please enter at least one URL")
        st.stop()

    if not question.strip():
        st.warning("⚠️ Please enter a question")
        st.stop()

    try:
        start_time = time.time()
        with st.spinner("Processing webpage and generating answer... 🤔"):

            urls_tuple = tuple(valid_urls)

            if urls_tuple == (active.get("current_url") or ()):
                retriever = active["current_retriever"]
                full_page_content = active["current_page_content"]
            else:
                retriever, full_page_content = create_rag_pipeline(urls_tuple)

                if retriever is None:
                    st.error("❌ Could not load any of the provided URLs.")
                    st.stop()

                active["current_url"] = urls_tuple
                active["current_retriever"] = retriever
                active["current_page_content"] = full_page_content

            # Track URL history
            for u in valid_urls:
                title = get_page_title(u)
                if title not in active["url_history"]:
                    active["url_history"][title] = u

            st.success("Website(s) loaded successfully ✅")

            retrieved_docs = retriever.invoke(question)

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )

            if summarize_clicked:
                answer = llm.invoke(
                    f"""
Summarize the following webpage content from multiple sources.

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
{full_page_content[:8000]}
"""
                ).content

            else:
                prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

The context may contain information from MULTIPLE websites.

When answering:
- Combine information from all relevant websites.
- If multiple websites discuss different aspects of the question, merge them into one complete answer.
- Mention differences when relevant.
- Do not ignore useful information from any source.
- Use all provided context.

If part of the question is answered by one website and another part by another website,
combine both answers.

Only say "Not found in the provided websites" when no relevant information exists in any source.

Context:
{context}

Question:
{question}

Answer:
""")

                def format_docs(docs):
                    return "\n\n".join(
                        f"[Source: {d.metadata.get('source_url', 'Unknown')}]\n{d.page_content}"
                        for d in docs
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

            response_time = round(time.time() - start_time, 2)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    active["messages"].append({
        "question": question,
        "answer": answer,
        "sources": retrieved_docs,
        "time": datetime.now().strftime("%I:%M %p"),
        "response_time": response_time
    })

# ─── Chat History Display ─────────────────────────────────────────────────────

for idx, msg in enumerate(active["messages"]):
    with st.chat_message("user"):
        st.caption(f"🕒 {msg['time']}")
        st.write(msg["question"])

    with st.chat_message("assistant"):
        st.caption(f"🕒 {msg['time']}")
        st.markdown(msg["answer"])

        word_count = len(msg["answer"].split())
        st.caption(f"📊 Words: {word_count} | Sources: {len(msg['sources'])} | ⏱ {msg['response_time']} sec")

        # Copy button using clipboard JS
        copy_key = f"copy_{st.session_state.active_chat}_{idx}"
        escaped = msg["answer"].replace("`", "\\`").replace("$", "\\$")
        copy_html = f"""
        <button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{
            this.textContent='✅ Copied!';
            setTimeout(()=>this.textContent='📋 Copy Answer',1500);
        }})" style="
            background:#2d2d2d;color:#fff;border:1px solid #555;
            padding:4px 12px;border-radius:6px;cursor:pointer;font-size:13px;
            margin-top:4px;
        ">📋 Copy Answer</button>
        """
        st.components.v1.html(copy_html, height=45)

        with st.expander("📚 Sources Used"):
            for i, doc in enumerate(msg["sources"], 1):
                source = doc.metadata.get("source_url", "Unknown Source")
                st.markdown(f"### Source {i}")
                st.caption(source)
                st.write(doc.page_content[:500])

# ─── Download Chat ────────────────────────────────────────────────────────────

chat_text = ""
for msg in active["messages"]:
    chat_text += f"User: {msg['question']}\n\nAI: {msg['answer']}\n\n" + "-" * 50 + "\n\n"

st.download_button(
    "⬇ Download Chat",
    chat_text,
    f"{active['name'].replace(' ', '_')}_history.txt"
)
