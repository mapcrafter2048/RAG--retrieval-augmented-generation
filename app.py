import os
import time
import faiss
import numpy as np
import PyPDF2
import requests
import speech_recognition as sr
import logging
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from evaluation import (
    compute_recall_at_k,
    compute_mrr,
    compute_average_relevancy,
    exact_match,
    compute_f1,
    compute_semantic_similarity
)

logging.basicConfig(level=logging.INFO)

HF_API_KEY = ""

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.jinja_env.globals.update(zip=zip)

BASE_KB_DIR = os.path.abspath("./knowledge_bases")
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
UPLOAD_FOLDER = BASE_KB_DIR

app.config['BASE_KB_DIR'] = BASE_KB_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CURRENT_KB'] = None

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Sentence Transformer model: {e}")
    embedder = None

faiss_index = None
chunk_mapping = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_safe_path(base, path_to_check):
    abs_base = os.path.abspath(base)
    abs_path = os.path.abspath(path_to_check)
    return os.path.commonpath([abs_base]) == os.path.commonpath([abs_base, abs_path])

def load_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
        return ""

def load_pdf_file(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_e:
                     logging.warning(f"Could not extract text from page {page_num + 1} in {file_path}: {page_e}")

    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return text

def load_documents(kb_directory):
    docs = []
    if not os.path.isdir(kb_directory) or not is_safe_path(app.config['BASE_KB_DIR'], kb_directory):
        logging.error(f"Invalid or unsafe knowledge base directory: {kb_directory}")
        return docs

    logging.info(f"Loading documents from: {kb_directory}")
    try:
        for filename in os.listdir(kb_directory):
            full_path = os.path.join(kb_directory, filename)
            if os.path.isfile(full_path) and is_safe_path(kb_directory, full_path):
                logging.debug(f"Processing file: {filename}")
                if filename.lower().endswith('.txt'):
                    docs.append({"text": load_txt_file(full_path), "source": filename})
                elif filename.lower().endswith('.pdf'):
                    pdf_text = load_pdf_file(full_path)
                    if pdf_text and pdf_text.strip():
                        docs.append({"text": pdf_text, "source": filename})
                    else:
                        logging.warning(f"No text extracted or empty PDF: {filename}")
    except Exception as e:
        logging.error(f"Error listing or processing directory {kb_directory}: {e}")

    logging.info(f"Loaded {len(docs)} documents.")
    return docs

def list_knowledge_bases(base_dir=None):
    if base_dir is None:
        base_dir = app.config['BASE_KB_DIR']
    kb_list = []
    if not os.path.exists(base_dir):
        logging.info(f"Base knowledge base directory not found, creating: {base_dir}")
        try:
            os.makedirs(base_dir)
        except OSError as e:
            logging.error(f"Failed to create base KB directory {base_dir}: {e}")
            return []

    try:
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and is_safe_path(base_dir, full_path):
                kb_list.append(full_path)
    except OSError as e:
        logging.error(f"Error listing knowledge bases in {base_dir}: {e}")
    return kb_list

def list_files_in_kb(kb_directory):
    files = []
    if not kb_directory or not os.path.isdir(kb_directory) or not is_safe_path(app.config['BASE_KB_DIR'], kb_directory):
        logging.warning(f"Cannot list files, invalid KB directory: {kb_directory}")
        return files
    try:
        for filename in os.listdir(kb_directory):
            full_path = os.path.join(kb_directory, filename)
            if os.path.isfile(full_path) and is_safe_path(kb_directory, full_path):
                files.append(filename)
    except OSError as e:
        logging.error(f"Error listing files in {kb_directory}: {e}")
    return sorted(files)

def speech_to_text_from_mic():
    pass

def store_text_in_kb(text, kb_directory):
    if not kb_directory or not os.path.isdir(kb_directory) or not is_safe_path(app.config['BASE_KB_DIR'], kb_directory):
        logging.error(f"Cannot store text, invalid or unsafe KB directory: {kb_directory}")
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_filename = f"speech_{timestamp}.txt"
    filename = os.path.join(kb_directory, base_filename)

    if not is_safe_path(kb_directory, filename):
         logging.error(f"Unsafe path generated for storing speech: {filename}")
         return None

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Stored speech transcript to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error writing speech transcript to {filename}: {e}")
        return None

def chunk_text(text, source_doc, chunk_size=300, overlap=50):
    if not text: return []
    words = text.split()
    chunks = []
    start = 0
    doc_len = len(words)
    while start < doc_len:
        end = min(start + chunk_size, doc_len)
        chunk_text_val = ' '.join(words[start:end])
        chunks.append({"text": chunk_text_val, "source": source_doc})
        next_start = start + chunk_size - overlap
        if next_start <= start: next_start = start + 1
        start = next_start
        if start >= doc_len and end < doc_len:
             last_chunk_text = ' '.join(words[start - (chunk_size - overlap) : doc_len])
             if last_chunk_text:
                 chunks.append({"text": last_chunk_text, "source": source_doc})
             break
    unique_chunks = []
    seen_texts = set()
    for chunk in chunks:
        ct = chunk["text"]
        if ct and ct not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(ct)
    return unique_chunks

def build_vector_store(doc_chunks):
    global faiss_index, chunk_mapping, embedder
    if embedder is None:
        logging.error("Cannot build vector store: Sentence Transformer model is not loaded.")
        faiss_index = None
        chunk_mapping = {}
        return False

    if not doc_chunks:
        logging.warning("No chunks provided to build vector store. Index will be empty.")
        faiss_index = None
        chunk_mapping = {}
        return True

    chunk_texts = [chunk['text'] for chunk in doc_chunks]

    try:
        logging.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        start_time = time.time()
        embeddings = embedder.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        end_time = time.time()
        logging.info(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

        if embeddings is None or embeddings.size == 0:
             logging.error("Embeddings generation failed or produced empty result.")
             faiss_index = None
             chunk_mapping = {}
             return False

        d = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(embeddings)
        chunk_mapping = { i: chunk for i, chunk in enumerate(doc_chunks) }
        logging.info(f"Built FAISS index with {faiss_index.ntotal} vectors from {len(doc_chunks)} chunks.")
        return True

    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        faiss_index = None
        chunk_mapping = {}
        return False

def retrieve_relevant_chunks(query, k=3):
    global faiss_index, chunk_mapping, embedder
    if embedder is None:
        logging.error("Cannot retrieve chunks: Sentence Transformer model is not loaded.")
        return []
    if faiss_index is None or not chunk_mapping:
        logging.warning("FAISS index is not initialized or empty. Cannot retrieve chunks.")
        return []

    if faiss_index.ntotal == 0:
        logging.debug("FAISS index is empty, cannot retrieve.")
        return []

    try:
        query_emb = embedder.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_emb, k)
        retrieved = [chunk_mapping[i] for i in indices[0] if i in chunk_mapping]
        logging.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
        return retrieved
    except Exception as e:
        logging.error(f"Error retrieving chunks from FAISS: {e}")
        return []

def rebuild_index(kb_path):
    global faiss_index, chunk_mapping
    if not kb_path or not os.path.isdir(kb_path) or not is_safe_path(app.config['BASE_KB_DIR'], kb_path):
         flash(f"Cannot rebuild index: Invalid or unsafe KB path '{kb_path}'.", "error")
         app.config['CURRENT_KB'] = None
         faiss_index = None
         chunk_mapping = {}
         return False

    logging.info(f"Rebuilding index for: {kb_path}")
    docs = load_documents(kb_path)
    if not docs:
        logging.warning(f"No documents found in '{os.path.basename(kb_path)}'. Index will be empty.")
        all_chunks = []
    else:
        all_chunks = []
        for doc in docs:
            if doc.get("text") and doc.get("source"):
                 chunks = chunk_text(doc["text"], source_doc=doc["source"])
                 all_chunks.extend(chunks)
            else:
                 logging.warning(f"Skipping document with missing text or source: {doc.get('source', 'N/A')}")

    success = build_vector_store(all_chunks)
    if success:
        app.config['CURRENT_KB'] = kb_path
        logging.info(f"Index rebuild completed for '{os.path.basename(kb_path)}'.")
        return True
    else:
        flash(f"Failed to rebuild vector store for '{os.path.basename(kb_path)}'.", "error")
        app.config['CURRENT_KB'] = None
        faiss_index = None
        chunk_mapping = {}
        return False

def construct_prompt(query, context_chunks):
    if not context_chunks: context_text = "No relevant context found."
    else:
        context_items = []
        for chunk in context_chunks:
            source = chunk.get("source", "Unknown")
            context_items.append(f"Source: {source}\nContent: {chunk['text']}")
        context_text = "\n\n---\n\n".join(context_items)
    prompt = (
        "You are a helpful assistant..."
        f"Context:\n{context_text}\n\n"
        "---\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )
    return prompt

def generate_response(prompt, model_name, max_tokens=250):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = { "inputs": prompt, "parameters": { "max_new_tokens": max_tokens, "temperature": 0.7, "top_p": 0.9, "do_sample": True, "return_full_text": False }, "options": { "wait_for_model": True } }
    logging.info(f"Sending request to HF Inference API for model: {model_name}")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        else: return f"Error: Unexpected API response format."
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        error_msg = f"Error: Failed to get response from {model_name}."
        return error_msg
    except Exception as e: return f"Error: An unexpected error occurred."

def evaluate_retrieval_simple(retrieved_chunks):
    if not retrieved_chunks: return {"Recall@3": 0.0, "MRR": 0.0}
    relevant_proxy = [retrieved_chunks[0]['text']]
    retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    recall = compute_recall_at_k(relevant_proxy, [retrieved_texts], k=3)
    mrr = compute_mrr(relevant_proxy, [retrieved_texts])
    return { "Recall@3": round(recall, 4), "MRR": round(mrr, 4) }

@app.route("/")
def index():
    kb_list = list_knowledge_bases()
    kb_display = [(os.path.basename(kb), kb) for kb in kb_list]
    current_kb_path = app.config.get('CURRENT_KB')
    current_kb_name = os.path.basename(current_kb_path) if current_kb_path else "None"
    return render_template("index.html", kb_list=kb_display, current_kb_name=current_kb_name)

@app.route("/set_kb", methods=["POST"])
def set_kb():
    global faiss_index, chunk_mapping
    selected_kb = request.form.get("knowledge_base")

    if not selected_kb or not os.path.isdir(selected_kb) or not is_safe_path(app.config['BASE_KB_DIR'], selected_kb):
        flash("Invalid knowledge base selected or directory not found.", "error")
        app.config['CURRENT_KB'] = None
        faiss_index = None
        chunk_mapping = {}
        return redirect(url_for("index"))

    success = rebuild_index(selected_kb)

    if success:
         num_chunks = len(chunk_mapping)
         num_vectors = faiss_index.ntotal if faiss_index else 0
         flash(f"Knowledge base '{os.path.basename(selected_kb)}' loaded with {num_chunks} chunks and {num_vectors} vectors.", "success")
    else:
        pass

    return redirect(url_for("query_page"))

@app.route("/create_kb", methods=["POST"])
def create_kb():
    kb_name = request.form.get("kb_name")
    if not kb_name:
        flash("Knowledge base name cannot be empty.", "warning")
        return redirect(url_for("index"))

    sanitized_name = "".join(c for c in kb_name if c.isalnum() or c in ('_', '-')).strip()
    if not sanitized_name:
         flash("Invalid characters in knowledge base name.", "error")
         return redirect(url_for("index"))

    new_kb_path = os.path.join(app.config['BASE_KB_DIR'], sanitized_name)

    if not is_safe_path(app.config['BASE_KB_DIR'], new_kb_path):
        flash("Potentially unsafe knowledge base name.", "error")
        return redirect(url_for("index"))

    if os.path.exists(new_kb_path):
        flash(f"Knowledge base '{sanitized_name}' already exists.", "warning")
    else:
        try:
            os.makedirs(new_kb_path)
            flash(f"Knowledge base '{sanitized_name}' created successfully.", "success")
            try:
                with open(os.path.join(new_kb_path, ".keep"), "w") as f: f.write("")
            except: pass
        except OSError as e:
            flash(f"Error creating knowledge base '{sanitized_name}': {e}", "error")

    return redirect(url_for("index"))

@app.route("/delete_kb", methods=["POST"])
def delete_kb():
    global faiss_index, chunk_mapping
    kb_path_to_delete = request.form.get("kb_path")

    if not kb_path_to_delete:
         flash("No knowledge base path provided for deletion.", "error")
         return redirect(url_for("index"))

    if not is_safe_path(app.config['BASE_KB_DIR'], kb_path_to_delete) or \
       os.path.abspath(kb_path_to_delete) == app.config['BASE_KB_DIR']:
        flash("Invalid or unsafe path provided for deletion.", "error")
        return redirect(url_for("index"))

    if not os.path.isdir(kb_path_to_delete):
         flash("Knowledge base directory not found.", "error")
         return redirect(url_for("index"))

    kb_name = os.path.basename(kb_path_to_delete)
    try:
        shutil.rmtree(kb_path_to_delete)
        flash(f"Knowledge base '{kb_name}' deleted successfully.", "success")

        current_kb = app.config.get('CURRENT_KB')
        if current_kb and os.path.abspath(current_kb) == os.path.abspath(kb_path_to_delete):
            app.config['CURRENT_KB'] = None
            faiss_index = None
            chunk_mapping = {}
            flash("Current knowledge base was deleted, resetting.", "info")

    except OSError as e:
        flash(f"Error deleting knowledge base '{kb_name}': {e}", "error")

    return redirect(url_for("index"))

@app.route("/query", methods=["GET", "POST"])
def query_page():
    current_kb_path = app.config.get('CURRENT_KB')

    if not current_kb_path:
        flash("Please select and load a knowledge base first.", "warning")
        return redirect(url_for("index"))

    if embedder is None:
        flash("Embedding model is not available. Cannot process queries.", "error")
        return render_template("query.html", current_kb_name=os.path.basename(current_kb_path), kb_files=[], model_error=True)

    query = None
    answer = None
    eval_metrics = None
    retrieved_context_for_display = None
    kb_files = list_files_in_kb(current_kb_path)

    if request.method == "POST":
        query = request.form.get("query")
        model_name = request.form.get("model")

        if not query: flash("Please enter a query.", "warning")
        elif not model_name: flash("Please select a model.", "warning")
        elif faiss_index is None or faiss_index.ntotal == 0:
            flash("The knowledge base index is empty or not built. Cannot retrieve context.", "warning")
        else:
            try:
                start_time = time.time()
                context_chunks = retrieve_relevant_chunks(query, k=3)
                retrieval_time = time.time() - start_time
                retrieved_context_for_display = [(chunk['text'], chunk.get('source', 'N/A')) for chunk in context_chunks]
                prompt = construct_prompt(query, context_chunks)

                start_time = time.time()
                answer = generate_response(prompt, model_name=model_name)
                generation_time = time.time() - start_time

                eval_metrics = evaluate_retrieval_simple(context_chunks)
                logging.info(f"Query: '{query[:50]}...'. Retrieval: {retrieval_time:.2f}s. Generation: {generation_time:.2f}s.")

            except Exception as e:
                logging.error(f"Error during query processing: {e}")
                flash(f"An unexpected error occurred during query: {e}", "error")

    current_kb_name = os.path.basename(current_kb_path) if current_kb_path else "None"
    return render_template("query.html",
                           query=query,
                           answer=answer,
                           eval_metrics=eval_metrics,
                           context_chunks=retrieved_context_for_display,
                           current_kb_name=current_kb_name,
                           kb_files=kb_files,
                           current_kb_path=current_kb_path
                           )

@app.route("/upload_file", methods=["POST"])
def upload_file():
    current_kb_path = app.config.get('CURRENT_KB')
    if not current_kb_path:
        flash("No knowledge base selected. Please select a KB first.", "warning")
        return redirect(url_for("index"))

    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('query_page'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file.', 'warning')
        return redirect(url_for('query_page'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(current_kb_path, filename)
        if not is_safe_path(current_kb_path, save_path):
             flash("Error: Detected potentially unsafe file path.", "error")
             return redirect(url_for('query_page'))

        try:
            file.save(save_path)
            flash(f"File '{filename}' uploaded successfully to '{os.path.basename(current_kb_path)}'.", "success")

            flash("Rebuilding index to include the new file...", "info")
            rebuild_success = rebuild_index(current_kb_path)
            if rebuild_success:
                flash("Index rebuilt successfully.", "success")

        except Exception as e:
             flash(f"Error saving file '{filename}': {e}", "error")

    else:
        flash('Invalid file type. Only .txt and .pdf allowed.', 'warning')

    return redirect(url_for('query_page'))

@app.route("/delete_file", methods=["POST"])
def delete_file():
    filename_to_delete = request.form.get("filename")
    kb_path_of_file = request.form.get("kb_path")

    if not filename_to_delete or not kb_path_of_file:
        flash("Missing filename or knowledge base path for deletion.", "error")
        return redirect(request.referrer or url_for('query_page'))

    filename_to_delete = secure_filename(filename_to_delete)

    if not is_safe_path(app.config['BASE_KB_DIR'], kb_path_of_file) or not os.path.isdir(kb_path_of_file):
         flash("Invalid or unsafe knowledge base path provided.", "error")
         return redirect(request.referrer or url_for('query_page'))

    full_file_path = os.path.join(kb_path_of_file, filename_to_delete)
    if not is_safe_path(kb_path_of_file, full_file_path) or not os.path.isfile(full_file_path):
         flash(f"File '{filename_to_delete}' not found or unsafe path.", "error")
         return redirect(request.referrer or url_for('query_page'))

    try:
        os.remove(full_file_path)
        flash(f"File '{filename_to_delete}' deleted successfully.", "success")

        current_kb_path = app.config.get('CURRENT_KB')
        if current_kb_path and os.path.abspath(current_kb_path) == os.path.abspath(kb_path_of_file):
            flash("Rebuilding index after file deletion...", "info")
            rebuild_success = rebuild_index(kb_path_of_file)
            if rebuild_success:
                flash("Index rebuilt successfully.", "success")
        else:
             flash("Index not rebuilt as the deleted file was not in the currently loaded KB.", "info")

    except OSError as e:
        flash(f"Error deleting file '{filename_to_delete}': {e}", "error")

    return redirect(url_for('query_page'))

@app.route("/speech", methods=["GET", "POST"])
def speech_input():
    current_kb_path = app.config.get('CURRENT_KB')
    if not current_kb_path:
        flash("No knowledge base selected. Please select a KB first.", "warning")
        return redirect(url_for("index"))

    if request.method == "POST":
        transcript = request.form.get("transcript")
        if not transcript or not transcript.strip():
            flash("No transcript received or transcript is empty.", "warning")
            return redirect(url_for("speech_input"))

        filename = store_text_in_kb(transcript, current_kb_path)
        if filename:
            flash(f"Speech transcript saved to '{os.path.basename(filename)}'. Rebuilding index...", "success")
            rebuild_success = rebuild_index(current_kb_path)
            if rebuild_success:
                flash("Index rebuilt successfully.", "success")
            return redirect(url_for("query_page"))
        else:
            flash("Failed to save speech transcript.", "error")
            return redirect(url_for("speech_input"))

    current_kb_name = os.path.basename(current_kb_path) if current_kb_path else "None"
    return render_template("speech.html", current_kb_name=current_kb_name)

@app.route("/rag_eval", methods=["GET", "POST"])
def rag_eval():
    current_kb_path = app.config.get('CURRENT_KB')
    results = None
    eval_data = {}

    if not current_kb_path:
        flash("Please select and load a knowledge base first to run evaluation.", "warning")
        return redirect(url_for("index"))

    if embedder is None:
        flash("Embedding model is not available. Cannot run evaluation.", "error")
        return render_template("rag_eval.html", results=None, current_kb_name="N/A", error="Embedding model not loaded.")

    if faiss_index is None or faiss_index.ntotal == 0:
        flash("The knowledge base index is empty or not built. Cannot run evaluation.", "warning")
        return render_template("rag_eval.html", results=None, current_kb_name=os.path.basename(current_kb_path), error="KB index empty.")

    if request.method == "POST":
        question = request.form.get("question")
        ground_truth = request.form.get("ground_truth")
        model_name = request.form.get("model")

        if not question or not ground_truth or not model_name:
            flash("Please fill in all fields: Question, Ground Truth Answer, and select a Model.", "warning")
        else:
            try:
                eval_data['question'] = question
                eval_data['ground_truth'] = ground_truth
                eval_data['model_name'] = model_name

                context_chunks_dicts = retrieve_relevant_chunks(question, k=3)
                context_chunks_texts = [chunk['text'] for chunk in context_chunks_dicts]
                eval_data['retrieved_context'] = [(chunk['text'], chunk.get('source', 'N/A')) for chunk in context_chunks_dicts]

                prompt = construct_prompt(question, context_chunks_dicts)
                generated_answer = generate_response(prompt, model_name=model_name)
                eval_data['generated_answer'] = generated_answer

                query_emb = embedder.encode([question], convert_to_numpy=True)
                chunk_embs = embedder.encode(context_chunks_texts, convert_to_numpy=True) if context_chunks_texts else np.array([])
                answer_emb = embedder.encode([generated_answer], convert_to_numpy=True)
                truth_emb = embedder.encode([ground_truth], convert_to_numpy=True)

                if context_chunks_texts:
                    relevant_proxy = [context_chunks_texts[0]]
                    recall = compute_recall_at_k(relevant_proxy, [context_chunks_texts], k=3)
                    mrr = compute_mrr(relevant_proxy, [context_chunks_texts])
                    relevancy = compute_average_relevancy(query_emb, chunk_embs)
                else: recall, mrr, relevancy = 0.0, 0.0, 0.0

                em = exact_match(generated_answer, ground_truth)
                f1 = compute_f1(generated_answer, ground_truth)
                sim = compute_semantic_similarity(answer_emb, truth_emb) if answer_emb.size > 0 and truth_emb.size > 0 else 0.0

                results = {
                    "Question": question, "Ground_Truth": ground_truth, "Generated_Answer": generated_answer,
                    "Retrieved_Context": eval_data['retrieved_context'], "Recall@3": round(recall, 4),
                    "MRR": round(mrr, 4), "Context_Relevancy": round(relevancy, 4), "Exact_Match": em,
                    "F1": round(f1, 4), "Answer_Similarity": round(sim, 4)
                }

            except Exception as e:
                logging.error(f"Error during RAG evaluation: {e}")
                flash(f"An unexpected error occurred during evaluation: {e}", "error")

    current_kb_name = os.path.basename(current_kb_path) if current_kb_path else "None"
    return render_template("rag_eval.html", results=results, current_kb_name=current_kb_name)

if __name__ == "__main__":
    if not os.path.exists(app.config['BASE_KB_DIR']):
        try:
            os.makedirs(app.config['BASE_KB_DIR'])
            logging.info(f"Created base knowledge base directory: {app.config['BASE_KB_DIR']}")
            default_kb_path = os.path.join(app.config['BASE_KB_DIR'], "Default")
            if not os.path.exists(default_kb_path):
                os.makedirs(default_kb_path)
                logging.info(f"Created default knowledge base directory: {default_kb_path}")
                try:
                    with open(os.path.join(default_kb_path, "knowledge.txt"), "w") as f:
                        f.write("This is a default knowledge base file.")
                except Exception as e: pass
        except OSError as e:
             logging.critical(f"CRITICAL: Failed to create base KB directory {app.config['BASE_KB_DIR']}. Exiting. Error: {e}")
             exit(1)

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)