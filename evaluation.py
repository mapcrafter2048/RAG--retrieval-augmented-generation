from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import Counter

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))

def compute_f1(pred, truth):
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_semantic_similarity(pred_embedding, truth_embedding):
    if pred_embedding is None or truth_embedding is None or pred_embedding.size == 0 or truth_embedding.size == 0:
        return 0.0
    pred_emb_2d = pred_embedding.reshape(1, -1)
    truth_emb_2d = truth_embedding.reshape(1, -1)
    similarity = cosine_similarity(pred_emb_2d, truth_emb_2d)
    return float(similarity[0][0])

def compute_recall_at_k(relevant_chunks_texts, retrieved_chunks_texts_list, k=3):
    if not relevant_chunks_texts:
        return 0.0

    correct = 0
    num_queries = len(relevant_chunks_texts)
    if num_queries != len(retrieved_chunks_texts_list):
        raise ValueError("Number of relevant chunks must match number of retrieved chunk lists.")

    for i in range(num_queries):
        true_relevant_text = relevant_chunks_texts[i]
        retrieved_for_query = retrieved_chunks_texts_list[i][:k]
        if true_relevant_text in retrieved_for_query:
            correct += 1

    return correct / num_queries

def compute_mrr(relevant_chunks_texts, retrieved_chunks_texts_list):
    if not relevant_chunks_texts:
        return 0.0

    reciprocal_ranks = []
    num_queries = len(relevant_chunks_texts)
    if num_queries != len(retrieved_chunks_texts_list):
        raise ValueError("Number of relevant chunks must match number of retrieved chunk lists.")

    for i in range(num_queries):
        true_relevant_text = relevant_chunks_texts[i]
        retrieved_for_query = retrieved_chunks_texts_list[i]
        try:
            rank = retrieved_for_query.index(true_relevant_text) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def compute_average_relevancy(query_embedding, chunk_embeddings):
    if query_embedding is None or chunk_embeddings is None or query_embedding.size == 0 or chunk_embeddings.size == 0:
        return 0.0

    query_emb_2d = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_emb_2d, chunk_embeddings)

    if similarities.size > 0:
        return float(np.mean(similarities[0]))
    else:
        return 0.0