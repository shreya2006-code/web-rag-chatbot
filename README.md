# Web RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using:

- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM
- Wikipedia Web Loader

## Features

- Chat with any webpage
- Semantic search using embeddings
- Vector database retrieval
- LLM-powered answers

## Run

```bash
pip install -r requirements.txt
python app.py
```

## Example

Ask:

- What is the main purpose of an operating system?
- What are the different types of operating systems?

The chatbot answers using only the webpage content.