# RAG-based Question Answering System

A Flask-based Question Answering system using Retrieval Augmented Generation (RAG) with document management capabilities.

## Features

- Document Management (PDF & TXT support)
- Knowledge Base Organization
- Vector Search using FAISS
- Speech-to-Text Input
- RAG Performance Evaluation
- Interactive Web Interface

## Setup

1. Clone the repository
2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face API key in app.py

## Usage

1. Run the application:

```bash
python app.py
```

2. Access the web interface at http://localhost:5000

3. Create a knowledge base and upload documents

4. Start asking questions!

## Evaluation Metrics

The system provides several evaluation metrics:

- Exact Match
- F1 Score
- Semantic Similarity
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context Relevancy

## Project Structure

- `app.py`: Main Flask application
- `evaluation.py`: Evaluation metrics implementation
- `templates/`: HTML templates
- `knowledge_bases/`: Document storage
