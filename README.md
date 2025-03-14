# Flask RAG Chatbot

This project is a Flask-based Retrieval-Augmented Generation (RAG) chatbot. It leverages a FAISS vector store, SentenceTransformer for embedding, and the Hugging Face Inference API to generate responses based on context from loaded knowledge bases.

## Features

- Load TXT and PDF documents as knowledge bases.
- Convert speech to text and add to a knowledge base.
- Query using various language models.
- Retrieve relevant context using FAISS.
