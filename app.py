import os
import time
import faiss
import numpy as np
import PyPDF2
import requests
import speech_recognition as sr
from flask import Flask, render_template, request, redirect, url_for, flash
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

HF_API_KEY = "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"

app.jinja_env.globals.update(zip=zip)

# Global configuration variables
BASE_KB_DIR = "./knowledge_bases"

# Initialize the embedding model.
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for the FAISS index and chunk mapping.
index = None
chunk_mapping = {}

###############################
# Data Ingestion Functions
###############################

def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf_file(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def load_documents(kb_directory):
    """Loads both TXT and PDF files from a given knowledge base directory."""
    docs = []
    for filename in os.listdir(kb_directory):
        full_path = os.path.join(kb_directory, filename)
        if os.path.isfile(full_path):
            if filename.lower().endswith('.txt'):
                docs.append(load_txt_file(full_path))
            elif filename.lower().endswith('.pdf'):
                pdf_text = load_pdf_file(full_path)
                if pdf_text.strip():
                    docs.append(pdf_text)
    return docs

###############################
# Knowledge Base Management
###############################

def list_knowledge_bases(base_dir=BASE_KB_DIR):
    """Lists subdirectories under the base directory (each is a knowledge base)."""
    kb_list = []
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            kb_list.append(full_path)
    return kb_list

###############################
# Speech-to-Text Functions
###############################

def speech_to_text():
    """Captures microphone input and converts it to text using Google’s Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Request error: {e}")
        return ""

def store_text_in_kb(text, kb_directory):
    """Stores transcribed speech as a new TXT file in the selected knowledge base."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(kb_directory, f"speech_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

###############################
# Text Chunking & Vector Store
###############################

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks to preserve context.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_vector_store(chunks):
    """Builds a FAISS index from text chunks."""
    global index, chunk_mapping
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
    print(f"Built FAISS index with {index.ntotal} vectors.")

def retrieve_relevant_chunks(query, k=3):
    """
    Retrieves the top k most relevant text chunks for a given query.
    Filters out any invalid indices.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    valid_indices = [idx for idx in indices[0] if idx != -1]
    retrieved_chunks = [chunk_mapping[idx] for idx in valid_indices]
    return retrieved_chunks

###############################
# Prompt & Response Generation
###############################

def construct_prompt(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful and knowledgeable assistant. "
        "Based on the following context, please answer the question below:\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )
    return prompt

def generate_response(prompt, model_name, max_tokens=150):
    """
    Generates a response using the Hugging Face Inference API for the specified model.
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    else:
        return f"Error: {response.json()}"

def answer_query(query, model_name):
    context = retrieve_relevant_chunks(query, k=3)
    prompt = construct_prompt(query, context)
    answer = generate_response(prompt, model_name=model_name)
    return answer

###############################
# Flask Routes & Views
###############################

@app.route("/")
def index():
    # List available knowledge bases
    kb_list = list_knowledge_bases()
    kb_display = [os.path.basename(kb) for kb in kb_list]
    return render_template("index.html", kb_list=kb_list, kb_display=kb_display)

@app.route("/set_kb", methods=["POST"])
def set_kb():
    selected_kb = request.form.get("knowledge_base")
    if not selected_kb or not os.path.exists(selected_kb):
        flash("Invalid knowledge base selected.")
        return redirect(url_for("index"))
    docs = load_documents(selected_kb)
    if not docs:
        flash("No documents found in the selected knowledge base.")
        return redirect(url_for("index"))
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))
    build_vector_store(chunks)
    app.config["CURRENT_KB"] = selected_kb
    flash(f"Knowledge base '{os.path.basename(selected_kb)}' loaded with {len(chunks)} chunks.")
    return redirect(url_for("query_page"))

@app.route("/query", methods=["GET", "POST"])
def query_page():
    if request.method == "POST":
        query = request.form.get("query")
        if not query:
            flash("Please enter a query.")
            return redirect(url_for("query_page"))
        # Get the selected model from the drop down.
        model_name = request.form.get("model")
        answer = answer_query(query, model_name)
        return render_template("query.html", query=query, answer=answer)
    else:
        return render_template("query.html")

@app.route("/speech", methods=["GET", "POST"])
def speech_input():
    if request.method == "POST":
        transcript = request.form.get("transcript")
        if not transcript:
            flash("No transcript received.")
            return redirect(url_for("speech_input"))
        selected_kb = app.config.get("CURRENT_KB")
        if not selected_kb:
            flash("No knowledge base selected. Please select a knowledge base first.")
            return redirect(url_for("index"))
        filename = store_text_in_kb(transcript, selected_kb)
        flash(f"Speech input stored in {filename}.")
        return redirect(url_for("query_page"))
    else:
        return render_template("speech.html")

if __name__ == "__main__":
    app.run(debug=True)
