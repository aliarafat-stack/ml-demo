import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "online-retail.csv")


@st.cache_data
def load_raw_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


@st.cache_data
def load_clean_transactions() -> pd.DataFrame:
    df = load_raw_transactions()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df.reset_index(drop=True)


@st.cache_data
def build_interaction_matrix():
    """Build user-item interaction matrix from purchase data.

    Returns:
        interactions: DataFrame (users × items) with purchase counts
        user_map: dict mapping user_idx -> CustomerID
        item_map: dict mapping item_idx -> StockCode
        item_desc: dict mapping StockCode -> Description
    """
    df = load_clean_transactions()

    user_item = (
        df.groupby(["CustomerID", "StockCode"])
        .agg(purchase_count=("Quantity", "sum"), description=("Description", "first"))
        .reset_index()
    )

    item_desc = dict(zip(user_item["StockCode"], user_item["description"]))

    users = user_item["CustomerID"].unique()
    items = user_item["StockCode"].unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    user_map = {i: u for u, i in user_to_idx.items()}
    item_map = {i: it for it, i in item_to_idx.items()}

    rows = user_item["CustomerID"].map(user_to_idx).values
    cols = user_item["StockCode"].map(item_to_idx).values
    vals = user_item["purchase_count"].values.astype(np.float32)

    interaction_sparse = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

    interactions = pd.DataFrame(
        interaction_sparse.toarray(),
        index=[user_map[i] for i in range(len(users))],
        columns=[item_map[i] for i in range(len(items))],
    )
    interactions.index.name = "CustomerID"

    return interactions, user_map, item_map, item_desc, interaction_sparse


def build_matrix_from_transactions(df, all_users, all_items):
    """Build a sparse interaction matrix from a subset of transactions using a fixed user/item universe."""
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {it: i for i, it in enumerate(all_items)}

    user_item = (
        df.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )

    valid = user_item[
        user_item["CustomerID"].isin(user_to_idx) & user_item["StockCode"].isin(item_to_idx)
    ]

    rows = valid["CustomerID"].map(user_to_idx).values
    cols = valid["StockCode"].map(item_to_idx).values
    vals = valid["Quantity"].values.astype(np.float32)

    return csr_matrix((vals, (rows, cols)), shape=(len(all_users), len(all_items)))


@st.cache_data
def get_rec_train_test(test_ratio: float = 0.2, random_state: int = 42):
    """Split interaction matrix into train/test by masking a fraction of each user's interactions."""
    interactions, user_map, item_map, item_desc, _ = build_interaction_matrix()
    rng = np.random.RandomState(random_state)

    train = interactions.copy()
    test_entries = []

    for user_idx in range(len(interactions)):
        row = interactions.iloc[user_idx]
        nonzero = row[row > 0].index.tolist()
        if len(nonzero) < 5:
            continue
        n_test = max(1, int(len(nonzero) * test_ratio))
        test_items = rng.choice(nonzero, size=n_test, replace=False)
        for item in test_items:
            test_entries.append((interactions.index[user_idx], item, row[item]))
            train.loc[train.index[user_idx], item] = 0

    test_df = pd.DataFrame(test_entries, columns=["CustomerID", "StockCode", "score"])

    train_sparse = csr_matrix(train.values.astype(np.float32))
    return train, train_sparse, test_df, user_map, item_map, item_desc
