import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares


@st.cache_resource
def train_svd(_train_sparse, n_factors: int = 50):
    """Truncated SVD on the user-item matrix."""
    k = min(n_factors, min(_train_sparse.shape) - 1)
    U, sigma, Vt = svds(_train_sparse.astype(float), k=k)
    sigma_diag = np.diag(sigma)
    predicted = U @ sigma_diag @ Vt
    return predicted, U, sigma, Vt


@st.cache_resource
def train_als(_train_sparse, n_factors: int = 50, iterations: int = 15, regularization: float = 0.1):
    """ALS for implicit feedback."""
    model = AlternatingLeastSquares(
        factors=n_factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
    )
    model.fit(_train_sparse)
    return model


@st.cache_resource
def train_nmf(_train_dense, n_factors: int = 50, max_iter: int = 200):
    """Non-negative Matrix Factorization."""
    model = NMF(n_components=n_factors, init="nndsvda", random_state=42, max_iter=max_iter)
    W = model.fit_transform(_train_dense)
    H = model.components_
    predicted = W @ H
    return predicted, W, H, model


@st.cache_resource
def train_item_cf(_train_sparse, top_k_similar: int = 20):
    """Item-based collaborative filtering using cosine similarity."""
    item_sim = cosine_similarity(_train_sparse.T)
    np.fill_diagonal(item_sim, 0)

    for i in range(item_sim.shape[0]):
        row = item_sim[i]
        threshold = np.partition(row, -top_k_similar)[-top_k_similar] if len(row) > top_k_similar else 0
        row[row < threshold] = 0
    item_sim_sparse = item_sim

    train_dense = _train_sparse.toarray() if hasattr(_train_sparse, 'toarray') else np.array(_train_sparse)
    predicted = train_dense @ item_sim_sparse
    return predicted, item_sim_sparse


def get_top_n_recommendations(predicted_scores, train_matrix, user_idx: int, n: int = 10):
    """Get top-N item indices for a user, excluding already-purchased items."""
    scores = predicted_scores[user_idx].copy()
    if hasattr(train_matrix, 'toarray'):
        purchased = train_matrix.toarray()[user_idx]
    else:
        purchased = np.array(train_matrix.iloc[user_idx] if hasattr(train_matrix, 'iloc') else train_matrix[user_idx])
    scores[purchased > 0] = -np.inf
    top_items = np.argsort(scores)[::-1][:n]
    return top_items, scores[top_items]


def evaluate_recommendations(predicted_scores, train_matrix, test_df, user_index, item_columns, k: int = 10):
    """Compute Precision@K, Recall@K, and Hit Rate across test users."""
    if hasattr(train_matrix, 'toarray'):
        train_dense = train_matrix.toarray()
    else:
        train_dense = np.array(train_matrix)

    item_to_col = {item: i for i, item in enumerate(item_columns)}

    precisions = []
    recalls = []
    hits = 0
    n_users = 0

    for cust_id, group in test_df.groupby("CustomerID"):
        if cust_id not in user_index:
            continue
        user_idx = user_index[cust_id]
        if user_idx >= predicted_scores.shape[0]:
            continue

        true_items = set()
        for _, row in group.iterrows():
            if row["StockCode"] in item_to_col:
                true_items.add(item_to_col[row["StockCode"]])
        if not true_items:
            continue

        top_items, _ = get_top_n_recommendations(predicted_scores, train_dense, user_idx, n=k)

        recommended_set = set(top_items)
        hit_count = len(recommended_set & true_items)

        precisions.append(hit_count / k)
        recalls.append(hit_count / len(true_items) if true_items else 0)
        if hit_count > 0:
            hits += 1
        n_users += 1

    if n_users == 0:
        return {"Precision@K": 0, "Recall@K": 0, "Hit Rate": 0, "Users Evaluated": 0}

    return {
        "Precision@K": float(np.mean(precisions)),
        "Recall@K": float(np.mean(recalls)),
        "Hit Rate": hits / n_users,
        "Users Evaluated": n_users,
    }
