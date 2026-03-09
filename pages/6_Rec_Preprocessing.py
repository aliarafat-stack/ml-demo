import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import csr_matrix

from utils.rec_data_loader import (
    load_raw_transactions, load_clean_transactions,
    build_interaction_matrix, get_rec_train_test,
)

st.set_page_config(page_title="Rec Preprocessing", page_icon="🔧", layout="wide")
st.title("Product Recommendation — Data Preprocessing")
st.markdown("---")

tab_clean, tab_matrix, tab_implicit, tab_split = st.tabs([
    "Data Cleaning", "Building the Matrix", "Implicit vs Explicit Feedback", "Train / Test Split"
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
with tab_clean:
    st.subheader("Step 1: Cleaning the Transaction Data")
    st.markdown("#### Why this matters")
    st.markdown(
        "Raw transaction data is messy. Cancelled orders, missing customer IDs, and "
        "test/admin entries would pollute our recommendations. We need to remove them "
        "before building the user-item matrix."
    )

    raw_df = load_raw_transactions()
    clean_df = load_clean_transactions()

    st.markdown("#### What we removed")

    cancelled = raw_df["InvoiceNo"].astype(str).str.startswith("C").sum()
    no_cust = raw_df["CustomerID"].isna().sum()
    neg_qty = (raw_df["Quantity"] <= 0).sum()
    zero_price = (raw_df["UnitPrice"] <= 0).sum()

    cleaning_df = pd.DataFrame({
        "Issue": ["Cancelled orders (InvoiceNo starts with 'C')",
                  "Missing CustomerID",
                  "Negative/zero quantity",
                  "Zero/negative price"],
        "Rows Affected": [f"{cancelled:,}", f"{no_cust:,}", f"{neg_qty:,}", f"{zero_price:,}"],
        "Why Remove": [
            "These are returns, not purchases — they'd confuse the model",
            "We can't recommend to anonymous users",
            "Indicates data errors or adjustments",
            "Free items / internal transfers, not real purchases",
        ],
    })
    st.dataframe(cleaning_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    before_c, arrow_c, after_c = st.columns([5, 1, 5])
    with before_c:
        st.metric("Before Cleaning", f"{len(raw_df):,} rows")
        st.dataframe(raw_df.head(8), use_container_width=True)
    with arrow_c:
        st.markdown("")
        st.markdown("")
        st.markdown("### →")
    with after_c:
        st.metric("After Cleaning", f"{len(clean_df):,} rows")
        st.dataframe(clean_df.head(8), use_container_width=True)

    st.markdown(
        f"We also added a `TotalPrice = Quantity × UnitPrice` column for revenue analysis."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Building the Matrix
# ═══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.subheader("Step 2: From Transactions to User-Item Matrix")
    st.markdown("#### Why this is the key step")
    st.markdown(
        "Recommendation algorithms don't work on raw transactions. They need a **matrix** "
        "where each row is a customer, each column is a product, and each cell tells us "
        "how much that customer interacted with that product."
    )

    st.graphviz_chart("""
        digraph matrix_build {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            txn [label="Transactions\\n(long format)\\nCustomer, Product,\\nQuantity per row", fillcolor="#dbeafe", color="#3b82f6"]
            agg [label="Aggregate\\nGroup by\\n(Customer, Product)\\nSum quantities", fillcolor="#e0e7ff", color="#6366f1"]
            pivot [label="Pivot to Matrix\\nRows = Customers\\nCols = Products\\nCells = Total Qty", fillcolor="#fce7f3", color="#ec4899"]
            sparse [label="Sparse Matrix\\nStore only non-zero\\ncells (saves memory)", fillcolor="#d1fae5", color="#10b981"]

            txn -> agg -> pivot -> sparse
        }
    """)

    clean_df = load_clean_transactions()

    st.markdown("#### Step-by-step transformation")

    st.markdown("**1. Raw transactions** — one row per line item:")
    st.dataframe(
        clean_df[["CustomerID", "StockCode", "Description", "Quantity"]].head(6),
        use_container_width=True,
    )

    st.markdown("**2. Aggregate** — sum quantities per (Customer, Product) pair:")
    agg_sample = (
        clean_df.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "TotalQuantity"})
    )
    st.dataframe(agg_sample.head(8), use_container_width=True)

    st.markdown("**3. Pivot to matrix** — customers as rows, products as columns:")

    interactions, user_map, item_map, item_desc, sparse_mat = build_interaction_matrix()
    n_users, n_items = interactions.shape

    sample_users = interactions.index[:5]
    sample_items = interactions.columns[:8]
    st.dataframe(interactions.loc[sample_users, sample_items], use_container_width=True)

    st.markdown(
        f"**Result:** {n_users:,} customers × {n_items:,} products = "
        f"{n_users * n_items:,.0f} cells. Most are zero (sparse)."
    )

    st.markdown("---")
    st.markdown("#### Why sparse format?")
    dense_size = n_users * n_items * 4
    nonzero = sparse_mat.nnz
    sparse_size = nonzero * (4 + 4 + 4)
    st.markdown(
        f"""
        | Format | Storage | Size |
        |---|---|---|
        | Dense matrix (store every cell) | {n_users:,} × {n_items:,} × 4 bytes | **{dense_size / 1024 / 1024:.1f} MB** |
        | Sparse matrix (store only non-zeros) | {nonzero:,} entries × 12 bytes | **{sparse_size / 1024 / 1024:.1f} MB** |
        | **Saving** | | **{(1 - sparse_size / dense_size):.0%} less memory** |

        With {(1 - nonzero / (n_users * n_items)):.2%} sparsity, storing all those zeros is wasteful.
        Sparse matrices (`scipy.sparse.csr_matrix`) store only the non-zero entries.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Implicit vs Explicit Feedback
# ═══════════════════════════════════════════════════════════════════════════════
with tab_implicit:
    st.subheader("Step 3: Understanding Implicit Feedback")

    st.markdown("#### Explicit vs Implicit — what's the difference?")
    st.markdown(
        """
        | | **Explicit Feedback** | **Implicit Feedback** (our data) |
        |---|---|---|
        | **What it is** | User directly rates an item (1-5 stars) | User behavior implies preference (purchase, click, view) |
        | **Example** | "I give this movie 4/5 stars" | "I bought this product 3 times" |
        | **Signal quality** | Clear — user told us directly | Ambiguous — did they buy it as a gift? |
        | **Negative signal** | 1-star = disliked | Absence of purchase ≠ disliked (maybe never saw it) |
        | **Volume** | Low (few users rate items) | High (every purchase is a data point) |
        | **Datasets** | MovieLens, Amazon Reviews | Our e-commerce data, Spotify plays, Netflix views |
        """
    )

    st.markdown("---")
    st.markdown("#### How we handle implicit feedback")
    st.markdown(
        """
        In our data, we have **purchase quantities** — not star ratings. A customer buying
        an item 5 times is a stronger signal than buying once. We use this as a
        **confidence weight**, not a rating:

        - **Purchase count = 0:** We don't know if the customer would like it (not necessarily negative)
        - **Purchase count = 1:** Some interest
        - **Purchase count = 5+:** Strong preference

        This distinction matters for the algorithms:
        - **SVD and NMF** treat the matrix values as scores to reconstruct
        - **ALS (Alternating Least Squares)** is specifically designed for implicit feedback —
          it distinguishes between "observed" and "unobserved" interactions
        """
    )

    st.success(
        "**Key insight:** Implicit feedback data is noisier than explicit ratings, "
        "but we have *much more* of it. Every purchase is a data point, whereas "
        "typically only 1-5% of users leave explicit ratings."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════════
with tab_split:
    st.subheader("Step 4: Train / Test Split for Recommendations")

    st.markdown("#### Why splitting is different here")
    st.markdown(
        "In classification (like churn), we split rows randomly. In recommendation, "
        "we split **interactions** — for each user, we hide some of their purchases "
        "and see if the model can predict them."
    )

    st.graphviz_chart("""
        digraph split {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            full [label="Full User-Item Matrix\\n(all interactions)", fillcolor="#dbeafe", color="#3b82f6"]
            train [label="Training Set (80%)\\nMost interactions visible", fillcolor="#d1fae5", color="#10b981"]
            test [label="Test Set (20%)\\nHidden interactions\\n(ground truth)", fillcolor="#fee2e2", color="#ef4444"]

            full -> train [label="Keep 80% of each\\nuser's purchases"]
            full -> test [label="Hide 20% of each\\nuser's purchases"]
        }
    """)

    train_df, train_sparse, test_df, _, _, _ = get_rec_train_test()

    t1, t2, t3 = st.columns(3)
    t1.metric("Training Interactions", f"{(train_df > 0).sum().sum():,}")
    t2.metric("Test Interactions", f"{len(test_df):,}")
    t3.metric("Test Users", f"{test_df['CustomerID'].nunique():,}")

    st.markdown(
        """
        **How we evaluate:** For each test user, we ask the model to recommend
        the top 10 items. We then check how many of those recommendations match
        the items we hid in the test set.

        | Metric | What it measures |
        |---|---|
        | **Precision@10** | Of the 10 recommended items, how many did the user actually buy? |
        | **Recall@10** | Of all hidden test items, how many appeared in the top 10? |
        | **Hit Rate** | What percentage of users got at least one correct recommendation? |
        """
    )

    st.warning(
        "**Note:** We only include users with ≥5 purchases in the evaluation, "
        "because users with very few interactions don't give the model enough signal "
        "to learn meaningful patterns."
    )
