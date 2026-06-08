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

# Load webpage
url = "https://en.wikipedia.org/wiki/Operating_system"

loader = WebBaseLoader(url)
docs = loader.load()

print(f"Loaded {len(docs)} document(s) from {url}")

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = Chroma.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.

If the answer is not present, say:
"Not found in the page."

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nChat with {url}")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    answer = chain.invoke(question)

    print(f"\nAI: {answer}\n")