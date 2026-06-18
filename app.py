from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

URLS = [
    "https://en.wikipedia.org/wiki/Operating_system",
    # Add more URLs here if needed
]

# ─── Load Documents from All URLs ─────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

all_docs = []

for url in URLS:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_url"] = url
    all_docs.extend(docs)
    print(f"✅ Loaded {len(docs)} doc(s) from {url}")

print(f"\nTotal documents loaded: {len(all_docs)}")

# ─── Split ────────────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks")

# ─── Vector Store ─────────────────────────────────────────────────────────────

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# ─── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ─── Prompt ───────────────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

The context may contain information from MULTIPLE websites.

When answering:
- Combine information from all relevant websites.
- If multiple websites discuss different aspects of the question, merge them into one complete answer.
- Mention differences when relevant.
- Do not ignore useful information from any source.
- Use all provided context.

Only say "Not found in the provided websites" when no relevant information exists in any source.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source_url', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

# ─── RAG Chain ────────────────────────────────────────────────────────────────

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ─── Chat Loop ────────────────────────────────────────────────────────────────

print(f"\nChat with: {', '.join(URLS)}")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"\nAI: {answer}\n")
