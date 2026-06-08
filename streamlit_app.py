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

load_dotenv()

st.title("🌐 Web RAG Chatbot")

url = st.text_input(
    "Enter Website URL",
    "https://en.wikipedia.org/wiki/Operating_system"
)

question = st.text_input("Ask a Question")

if st.button("Ask"):

    with st.spinner("Loading webpage..."):

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

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

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

    st.subheader("Answer")
    st.markdown(answer)