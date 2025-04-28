from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import f1_score # We implement f1 manually for token level
import numpy as np
import re
from collections import Counter

def normalize(text):
    """Lowercases, removes punctuation, and strips extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Keep only word characters and whitespace
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespace with single space
    return text

def exact_match(pred, truth):
    """Checks for exact string match after normalization."""
    return int(normalize(pred) == normalize(truth))

def compute_f1(pred, truth):
    """Computes the F1 score based on shared tokens after normalization."""
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0 # Return 0 if either prediction or truth is empty after normalization

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_semantic_similarity(pred_embedding, truth_embedding):
    """Computes cosine similarity between two embeddings."""
    if pred_embedding is None or truth_embedding is None or pred_embedding.size == 0 or truth_embedding.size == 0:
        return 0.0
    # Ensure embeddings are 2D arrays for cosine_similarity
    pred_emb_2d = pred_embedding.reshape(1, -1)
    truth_emb_2d = truth_embedding.reshape(1, -1)
    similarity = cosine_similarity(pred_emb_2d, truth_emb_2d)
    # cosine_similarity returns a 2D array, get the single value
    return float(similarity[0][0])

def compute_recall_at_k(relevant_chunks_texts, retrieved_chunks_texts_list, k=3):
    """
    Computes Recall@k.
    :param relevant_chunks_texts: A list of ground truth relevant chunk texts for each query.
    :param retrieved_chunks_texts_list: A list of lists, where each inner list contains the retrieved chunk texts for a query.
    :param k: The threshold for recall calculation.
    :return: Recall@k score.
    """
    if not relevant_chunks_texts:
        return 0.0

    correct = 0
    num_queries = len(relevant_chunks_texts)
    if num_queries != len(retrieved_chunks_texts_list):
        raise ValueError("Number of relevant chunks must match number of retrieved chunk lists.")

    for i in range(num_queries):
        true_relevant_text = relevant_chunks_texts[i]
        retrieved_for_query = retrieved_chunks_texts_list[i][:k] # Consider only top K retrieved
        # Check if the exact relevant text is within the top K retrieved texts
        if true_relevant_text in retrieved_for_query:
            correct += 1

    return correct / num_queries

def compute_mrr(relevant_chunks_texts, retrieved_chunks_texts_list):
    """
    Computes Mean Reciprocal Rank (MRR).
    :param relevant_chunks_texts: A list of ground truth relevant chunk texts for each query.
    :param retrieved_chunks_texts_list: A list of lists, where each inner list contains the retrieved chunk texts for a query.
    :return: MRR score.
    """
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
            # Find the rank (position) of the first occurrence of the relevant chunk
            rank = retrieved_for_query.index(true_relevant_text) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Relevant chunk was not found in the retrieved list
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks: # Avoid division by zero if list is empty
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_average_relevancy(query_embedding, chunk_embeddings):
    """
    Computes the average cosine similarity between the query embedding and retrieved chunk embeddings.
    Assumes higher similarity means higher relevancy.
    :param query_embedding: The embedding of the query (1, D).
    :param chunk_embeddings: A numpy array of embeddings for the retrieved chunks (N, D).
    :return: Average cosine similarity, or 0.0 if no chunk embeddings provided.
    """
    if query_embedding is None or chunk_embeddings is None or query_embedding.size == 0 or chunk_embeddings.size == 0:
        return 0.0

    # Ensure query embedding is 2D
    query_emb_2d = query_embedding.reshape(1, -1)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_emb_2d, chunk_embeddings)

    # similarities will be shape (1, N), so take the mean of the first (and only) row
    if similarities.size > 0:
        return float(np.mean(similarities[0]))
    else:
        return 0.0