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
import sqlite3
import hashlib
import smtplib
import random
import string
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Web RAG Chatbot",
    page_icon="🌐",
    layout="wide"
)

# ─── CSS Overrides ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Speed up transitions — remove Streamlit's default fade overlay */
.stApp > div:first-child { transition: none !important; }
div[data-testid="stToolbar"] { display: none; }

/* Fixed top header bar */
.top-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #0e1117;
    padding: 12px 0 6px 0;
    border-bottom: 1px solid #2d2d2d;
    margin-bottom: 10px;
}
.site-title {
    font-size: 2rem;
    font-weight: 800;
    color: #4da6ff;
    margin: 0;
    line-height: 1.1;
}
.chat-subtitle {
    font-size: 1rem;
    color: #aaa;
    margin: 2px 0 0 2px;
}

/* Chat rename input inline */
.rename-input input {
    font-size: 13px !important;
    padding: 2px 6px !important;
}

/* Thinking animation */
@keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }
.thinking-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4da6ff;
    margin: 0 2px;
    animation: blink 1.4s infinite both;
}
.thinking-dot:nth-child(2) { animation-delay: .2s; }
.thinking-dot:nth-child(3) { animation-delay: .4s; }
.thinking-box {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 16px;
    background: #1a1a2e;
    border-left: 3px solid #4da6ff;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 14px; color: #ccc;
}
</style>
""", unsafe_allow_html=True)

# ─── Embeddings (cached globally) ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ─── DB Setup ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "webrag_users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email_verified INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS otp_store (
            email TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            purpose TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chat_id TEXT NOT NULL,
            chat_name TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            url_history_json TEXT NOT NULL,
            website_urls_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, chat_id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, email, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (username.strip(), email.strip().lower(), hash_pw(password))
        )
        conn.commit()
        return True, "Registered successfully."
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        return False, "Email already registered."
    finally:
        conn.close()

def verify_login(username_or_email, password):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE (username=? OR email=?) AND password_hash=?",
        (username_or_email, username_or_email.lower(), hash_pw(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def set_email_verified(email):
    conn = get_db()
    conn.execute("UPDATE users SET email_verified=1 WHERE email=?", (email.lower(),))
    conn.commit()
    conn.close()

def update_password(email, new_password):
    conn = get_db()
    conn.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_pw(new_password), email.lower()))
    conn.commit()
    conn.close()

# ─── OTP ──────────────────────────────────────────────────────────────────────
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def store_otp(email, otp, purpose):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO otp_store (email, otp, purpose, expires_at) VALUES (?,?,?,?)",
        (email.lower(), otp, purpose, time.time() + 600)
    )
    conn.commit()
    conn.close()

def verify_otp(email, otp, purpose):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM otp_store WHERE email=? AND purpose=?",
        (email.lower(), purpose)
    ).fetchone()
    conn.close()
    if not row:
        return False, "No OTP found."
    if time.time() > row["expires_at"]:
        return False, "OTP expired."
    if row["otp"] != otp.strip():
        return False, "Invalid OTP."
    return True, "OK"

def send_otp_email(to_email, otp, purpose):
    """Send OTP via Gmail SMTP. Reads GMAIL_USER and GMAIL_APP_PASSWORD from .env"""
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_pass:
        return False, "Email sending not configured. Add GMAIL_USER and GMAIL_APP_PASSWORD to .env"
    subject = "Your Web RAG Chatbot OTP"
    body = f"""
Hello,

Your OTP for {purpose} is:

  {otp}

This code is valid for 10 minutes.

If you did not request this, ignore this email.

– Web RAG Chatbot
"""
    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())
        return True, "OTP sent."
    except Exception as e:
        return False, f"Email error: {e}"

# ─── Chat Persistence ─────────────────────────────────────────────────────────
def save_chats_to_db(user_id):
    """Persist all chats for logged-in user."""
    conn = get_db()
    for cid in st.session_state.chat_order:
        chat = st.session_state.chats[cid]
        # Don't serialize retriever/vectorstore objects
        msgs = []
        for m in chat["messages"]:
            msgs.append({
                "question": m["question"],
                "answer": m["answer"],
                "sources_text": [
                    {"content": d.page_content, "source_url": d.metadata.get("source_url", "")}
                    for d in m.get("sources", [])
                ],
                "time": m["time"],
                "response_time": m["response_time"]
            })
        conn.execute("""
            INSERT INTO chat_history
                (user_id, chat_id, chat_name, messages_json, url_history_json, website_urls_json, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(user_id, chat_id) DO UPDATE SET
                chat_name=excluded.chat_name,
                messages_json=excluded.messages_json,
                url_history_json=excluded.url_history_json,
                website_urls_json=excluded.website_urls_json,
                updated_at=excluded.updated_at
        """, (
            user_id, cid, chat["name"],
            json.dumps(msgs),
            json.dumps(chat.get("url_history", {})),
            json.dumps(chat.get("website_urls", [""])),
            datetime.now().isoformat()
        ))
    conn.commit()
    conn.close()

def load_chats_from_db(user_id):
    """Load saved chats for user; returns (chats_dict, chat_order_list)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM chat_history WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    chats = {}
    order = []
    for row in rows:
        cid = row["chat_id"]
        raw_msgs = json.loads(row["messages_json"])
        # Reconstruct lightweight message objects (sources as plain dicts)
        msgs = []
        for m in raw_msgs:
            msgs.append({
                "question": m["question"],
                "answer": m["answer"],
                "sources": [type("Doc", (), {"page_content": s["content"], "metadata": {"source_url": s["source_url"]}})() for s in m.get("sources_text", [])],
                "time": m["time"],
                "response_time": m["response_time"]
            })
        chats[cid] = {
            "name": row["chat_name"],
            "messages": msgs,
            "url_history": json.loads(row["url_history_json"]),
            "website_urls": json.loads(row["website_urls_json"]),
            "current_url": None,
            "current_retriever": None,
            "current_page_content": None,
        }
        order.append(cid)
    return chats, order

def delete_chat_from_db(user_id, chat_id):
    conn = get_db()
    conn.execute("DELETE FROM chat_history WHERE user_id=? AND chat_id=?", (user_id, chat_id))
    conn.commit()
    conn.close()

# ─── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("logged_in", False),
    ("user", None),
    ("auth_page", "login"),   # login | register | verify_email | forgot | reset_pw
    ("otp_email", ""),
    ("otp_purpose", ""),
    ("chats", None),
    ("chat_order", None),
    ("active_chat", None),
    ("search_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def init_chats_for_user(user_id):
    chats, order = load_chats_from_db(user_id)
    if chats:
        st.session_state.chats = chats
        st.session_state.chat_order = order
        st.session_state.active_chat = order[0]
    else:
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

# ─── Auth UI ──────────────────────────────────────────────────────────────────
def auth_ui():
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <div style="font-size:2.8rem; font-weight:900; color:#4da6ff;">🌐 Web RAG Chatbot</div>
        <div style="color:#888; margin-top:6px;">Chat with any webpage using AI</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.session_state.auth_page

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if page == "login":
            st.subheader("Sign In")
            u = st.text_input("Username or Email", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Sign In", use_container_width=True, type="primary"):
                user = verify_login(u, p)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    init_chats_for_user(user["id"])
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create account", use_container_width=True):
                    st.session_state.auth_page = "register"
                    st.rerun()
            with c2:
                if st.button("Forgot password", use_container_width=True):
                    st.session_state.auth_page = "forgot"
                    st.rerun()

        elif page == "register":
            st.subheader("Create Account")
            un = st.text_input("Username", key="reg_un")
            em = st.text_input("Email", key="reg_em")
            pw = st.text_input("Password", type="password", key="reg_pw")
            pw2 = st.text_input("Confirm Password", type="password", key="reg_pw2")
            if st.button("Register", use_container_width=True, type="primary"):
                if pw != pw2:
                    st.error("Passwords don't match.")
                elif len(pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = register_user(un, em, pw)
                    if ok:
                        otp = generate_otp()
                        store_otp(em, otp, "verify_email")
                        sent, smsg = send_otp_email(em, otp, "email verification")
                        st.session_state.otp_email = em
                        st.session_state.otp_purpose = "verify_email"
                        st.session_state.auth_page = "verify_email"
                        if not sent:
                            st.warning(f"Registered! Email send failed ({smsg}). OTP (dev mode): `{otp}`")
                        else:
                            st.success("Registered! Check your email for OTP.")
                        st.rerun()
                    else:
                        st.error(msg)
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()

        elif page == "verify_email":
            st.subheader("Verify Email")
            st.info(f"Enter the 6-digit OTP sent to **{st.session_state.otp_email}**")
            otp_in = st.text_input("OTP", max_chars=6, key="ve_otp")
            if st.button("Verify", use_container_width=True, type="primary"):
                ok, msg = verify_otp(st.session_state.otp_email, otp_in, "verify_email")
                if ok:
                    set_email_verified(st.session_state.otp_email)
                    st.success("Email verified! Please log in.")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(msg)
            if st.button("Resend OTP"):
                otp = generate_otp()
                store_otp(st.session_state.otp_email, otp, "verify_email")
                sent, smsg = send_otp_email(st.session_state.otp_email, otp, "email verification")
                if sent:
                    st.success("OTP resent.")
                else:
                    st.warning(f"Could not send email. Dev OTP: `{otp}`")

        elif page == "forgot":
            st.subheader("Forgot Password")
            em = st.text_input("Enter your registered email", key="fp_em")
            if st.button("Send OTP", use_container_width=True, type="primary"):
                user = get_user_by_email(em)
                if not user:
                    st.error("Email not found.")
                else:
                    otp = generate_otp()
                    store_otp(em, otp, "reset_pw")
                    sent, smsg = send_otp_email(em, otp, "password reset")
                    st.session_state.otp_email = em
                    st.session_state.otp_purpose = "reset_pw"
                    st.session_state.auth_page = "reset_pw"
                    if not sent:
                        st.warning(f"Email failed ({smsg}). Dev OTP: `{otp}`")
                    else:
                        st.success("OTP sent to your email.")
                    st.rerun()
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()

        elif page == "reset_pw":
            st.subheader("Reset Password")
            st.info(f"OTP sent to **{st.session_state.otp_email}**")
            otp_in = st.text_input("OTP", max_chars=6, key="rp_otp")
            new_pw = st.text_input("New Password", type="password", key="rp_pw")
            new_pw2 = st.text_input("Confirm New Password", type="password", key="rp_pw2")
            if st.button("Reset Password", use_container_width=True, type="primary"):
                if new_pw != new_pw2:
                    st.error("Passwords don't match.")
                elif len(new_pw) < 6:
                    st.error("Min 6 characters.")
                else:
                    ok, msg = verify_otp(st.session_state.otp_email, otp_in, "reset_pw")
                    if ok:
                        update_password(st.session_state.otp_email, new_pw)
                        st.success("Password reset! Please log in.")
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(msg)
            if st.button("Resend OTP"):
                otp = generate_otp()
                store_otp(st.session_state.otp_email, otp, "reset_pw")
                sent, smsg = send_otp_email(st.session_state.otp_email, otp, "password reset")
                if sent:
                    st.success("OTP resent.")
                else:
                    st.warning(f"Dev OTP: `{otp}`")

# ─── Show auth wall if not logged in ──────────────────────────────────────────
if not st.session_state.logged_in:
    auth_ui()
    st.stop()

# ─── Helpers (only after login) ───────────────────────────────────────────────
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
    save_chats_to_db(st.session_state.user["id"])

def delete_chat(cid):
    if len(st.session_state.chat_order) == 1:
        st.session_state.chats[cid] = {
            "name": "Chat 1",
            "messages": [],
            "url_history": {},
            "current_url": None,
            "current_retriever": None,
            "current_page_content": None,
            "website_urls": [""],
        }
        save_chats_to_db(st.session_state.user["id"])
        return
    delete_chat_from_db(st.session_state.user["id"], cid)
    st.session_state.chat_order.remove(cid)
    del st.session_state.chats[cid]
    st.session_state.active_chat = st.session_state.chat_order[-1]

@st.cache_data(show_spinner=False)
def get_page_title_cached(url):
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
    st.markdown(f"👤 **{st.session_state.user['username']}**")
    if st.button("🚪 Logout", use_container_width=True):
        save_chats_to_db(st.session_state.user["id"])
        for key in ["logged_in", "user", "chats", "chat_order", "active_chat"]:
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("**💬 Conversations**")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.session_state.search_query = st.text_input(
        "🔍 Search chats",
        value=st.session_state.search_query,
        placeholder="Search by name...",
        label_visibility="collapsed"
    )

    st.markdown("---")

    for cid in st.session_state.chat_order:
        chat = st.session_state.chats[cid]
        if st.session_state.search_query.lower() not in chat["name"].lower():
            continue
        is_active = cid == st.session_state.active_chat

        col_name, col_del = st.columns([6, 1])
        with col_name:
            # Inline rename: if active, show editable text_input; else show button
            if is_active:
                new_name = st.text_input(
                    "rename",
                    value=chat["name"],
                    key=f"rename_{cid}",
                    label_visibility="collapsed"
                )
                if new_name.strip() and new_name.strip() != chat["name"]:
                    st.session_state.chats[cid]["name"] = new_name.strip()
                    save_chats_to_db(st.session_state.user["id"])
            else:
                if st.button(chat["name"], key=f"sel_{cid}", use_container_width=True):
                    st.session_state.active_chat = cid
                    st.rerun()

        with col_del:
            st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
            if st.button("🗑", key=f"del_{cid}", help="Delete"):
                delete_chat(cid)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        get_active()["messages"] = []
        save_chats_to_db(st.session_state.user["id"])
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** llama-3.3-70b-versatile")
    st.info("Ask questions about any webpage using RAG.")

# ─── Main UI ──────────────────────────────────────────────────────────────────
active = get_active()

# Fixed top header
st.markdown(f"""
<div class="top-header">
    <div class="site-title">🌐 Web RAG Chatbot</div>
    <div class="chat-subtitle">💬 {active['name']}</div>
</div>
""", unsafe_allow_html=True)

# Recent websites dropdown
recent_site = st.selectbox(
    "🕒 Recent Websites",
    ["None"] + list(active["url_history"].keys()),
    label_visibility="visible"
)
selected_url = ""
if recent_site != "None":
    selected_url = active["url_history"][recent_site]

# ─── URL Inputs ───────────────────────────────────────────────────────────────
if "website_urls" not in active or not active["website_urls"]:
    active["website_urls"] = [""]

if selected_url and active["website_urls"][0] == "":
    active["website_urls"][0] = selected_url

urls_to_delete = []
updated_urls = []

st.markdown("### 🌐 Website URLs")

for i, url_val in enumerate(active["website_urls"]):
    col_input, col_del = st.columns([12, 1])
    with col_input:
        new_val = st.text_input(
            f"Website URL {i + 1}",
            value=url_val,
            key=f"url_input_{st.session_state.active_chat}_{i}"
        )
        updated_urls.append(new_val)
    with col_del:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        if len(active["website_urls"]) > 1:
            if st.button("✕", key=f"del_url_{st.session_state.active_chat}_{i}", help="Remove"):
                urls_to_delete.append(i)
        st.markdown("</div>", unsafe_allow_html=True)

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

# Current page info — one per URL
valid_filled = [u.strip() for u in active["website_urls"] if u.strip()]
if valid_filled:
    for u in valid_filled:
        title = get_page_title_cached(u)
        st.info(f"📄 Current Page: **{title}**  \n🔗 {u}")

# ─── Question & Buttons ───────────────────────────────────────────────────────
question = st.text_area("Ask a Question", height=100, placeholder="Type your question here…")

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("🔍 Ask", use_container_width=True, type="primary")
with col2:
    summarize_clicked = st.button("📝 Summarize Page", use_container_width=True)

st.caption("💡 Try: *What are the key points?* • *Explain like I'm a beginner* • *Summarize this page*")

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

    # Show thinking animation
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="thinking-box">
        <span>Thinking</span>
        <span class="thinking-dot"></span>
        <span class="thinking-dot"></span>
        <span class="thinking-dot"></span>
    </div>
    """, unsafe_allow_html=True)

    try:
        start_time = time.time()
        urls_tuple = tuple(valid_urls)

        if urls_tuple == (active.get("current_url") or ()):
            retriever = active["current_retriever"]
            full_page_content = active["current_page_content"]
        else:
            with st.spinner("Loading and indexing website(s)…"):
                retriever, full_page_content = create_rag_pipeline(urls_tuple)

            if retriever is None:
                thinking_placeholder.empty()
                st.error("❌ Could not load any of the provided URLs.")
                st.stop()

            active["current_url"] = urls_tuple
            active["current_retriever"] = retriever
            active["current_page_content"] = full_page_content

        # Track URL history
        for u in valid_urls:
            title = get_page_title_cached(u)
            if title not in active["url_history"]:
                active["url_history"][title] = u

        retrieved_docs = retriever.invoke(question)

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

        def format_docs(docs):
            return "\n\n".join(
                f"[Source: {d.metadata.get('source_url', 'Unknown')}]\n{d.page_content}"
                for d in docs
            )

        if summarize_clicked:
            answer = llm.invoke(f"""
Summarize the following webpage content from multiple sources.

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
""").content
        else:
            prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.
The context may contain information from MULTIPLE websites.

When answering:
- Combine information from all relevant websites.
- If multiple websites discuss different aspects, merge into one complete answer.
- Mention differences when relevant.
- Do not ignore useful information from any source.
- Use all provided context.

Only say "Not found in the provided websites" when no relevant information exists.

Context:
{context}

Question:
{question}

Answer:
""")
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            answer = chain.invoke(question)

        response_time = round(time.time() - start_time, 2)
        thinking_placeholder.empty()

    except Exception as e:
        thinking_placeholder.empty()
        st.error(f"❌ Error: {e}")
        st.stop()

    active["messages"].append({
        "question": question,
        "answer": answer,
        "sources": retrieved_docs,
        "time": datetime.now().strftime("%I:%M %p"),
        "response_time": response_time
    })
    save_chats_to_db(st.session_state.user["id"])

# ─── Chat History Display ─────────────────────────────────────────────────────
for idx, msg in enumerate(active["messages"]):
    with st.chat_message("user"):
        st.caption(f"🕒 {msg['time']}")
        st.write(msg["question"])

    with st.chat_message("assistant"):
        st.caption(f"🕒 {msg['time']}")

        # Collapsible answer
        with st.expander("💬 View Answer", expanded=True):
            st.markdown(msg["answer"])

            word_count = len(msg["answer"].split())
            st.caption(f"📊 Words: {word_count} | Sources: {len(msg['sources'])} | ⏱ {msg['response_time']} sec")

            # Copy button
            escaped = msg["answer"].replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace("\n", "\\n")
            copy_html = f"""
            <button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{
                this.textContent='✅ Copied!';
                setTimeout(()=>this.textContent='📋 Copy Answer',1500);
            }})" style="
                background:#1e3a5f;color:#9dc8ff;border:1px solid #3a6ea8;
                padding:5px 14px;border-radius:6px;cursor:pointer;font-size:13px;margin-top:6px;
            ">📋 Copy Answer</button>
            """
            st.components.v1.html(copy_html, height=48)

        with st.expander("📚 Sources Used"):
            for i, doc in enumerate(msg["sources"], 1):
                source = doc.metadata.get("source_url", "Unknown Source")
                st.markdown(f"**Source {i}:** `{source}`")
                st.write(doc.page_content[:500])
                st.markdown("---")

# ─── Download Chat ────────────────────────────────────────────────────────────
if active["messages"]:
    chat_text = ""
    for msg in active["messages"]:
        chat_text += f"User: {msg['question']}\n\nAI: {msg['answer']}\n\n" + "-"*50 + "\n\n"
    st.download_button(
        "⬇ Download Chat",
        chat_text,
        f"{active['name'].replace(' ', '_')}_history.txt"
    )
