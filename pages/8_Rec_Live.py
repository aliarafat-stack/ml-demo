import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from utils.rec_data_loader import load_clean_transactions, build_interaction_matrix, build_matrix_from_transactions

st.set_page_config(page_title="Rec Live Updates", page_icon="⚡", layout="wide")
st.title("Product Recommendations — Live Model Updates")
st.markdown("---")

tab_explain, tab_demo = st.tabs(["How It Works", "Live Demo"])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — How It Works
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("Keeping Recommendations Fresh")

    st.markdown(
        """
        Customer tastes change. New products arrive. Seasonal trends shift.
        A recommendation model trained on last year's data will miss all of this.
        We need **continuous updates** to stay relevant.
        """
    )

    st.graphviz_chart("""
        digraph rec_drift {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            jan [label="Jan\\nTrained on\\nholiday purchases\\nRecs: gift sets", fillcolor="#d1fae5", color="#10b981"]
            apr [label="Apr\\nSame model\\nSpring arrived\\nStill rec'ing gifts", fillcolor="#fef3c7", color="#f59e0b"]
            jul [label="Jul\\nSame model\\nSummer products\\nCompletely stale", fillcolor="#fee2e2", color="#ef4444"]

            jan -> apr [label="  drift"]
            apr -> jul [label="  more drift"]
        }
    """)

    st.markdown("---")
    st.subheader("Three Strategies for Updating Recommendation Models")

    st.markdown("#### Strategy 1: Full Retrain (Nightly)")
    st.graphviz_chart("""
        digraph strat1 {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            all [label="All historical\\npurchases", fillcolor="#dbeafe", color="#3b82f6"]
            retrain [label="Full ALS retrain\\n(minutes for our data)", fillcolor="#fce7f3", color="#ec4899"]
            deploy [label="Deploy new\\nmodel", fillcolor="#d1fae5", color="#10b981"]

            all -> retrain -> deploy
        }
    """)
    st.markdown(
        "**When to use:** Small-to-medium datasets, or as a weekly safety net. "
        "ALS trains fast — our 4,338 × 3,665 matrix trains in under a second."
    )

    st.markdown("#### Strategy 2: Incremental Factor Update (Real-time)")
    st.graphviz_chart("""
        digraph strat2 {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            existing [label="Existing ALS model\\n(user & item factors)", fillcolor="#d1fae5", color="#10b981"]
            new [label="New purchase\\nCustomer 12345\\nbought Cake Stand", fillcolor="#dbeafe", color="#3b82f6"]
            update [label="Recompute only\\nCustomer 12345's\\nuser factors", fillcolor="#fef3c7", color="#f59e0b"]
            live [label="Updated model\\n(milliseconds)", fillcolor="#d1fae5", color="#10b981"]

            existing -> update
            new -> update
            update -> live
        }
    """)
    st.markdown(
        "**When to use:** High-traffic systems where full retraining is too slow. "
        "Keep item factors fixed, recompute only the affected user's row. "
        "This is what Spotify does for real-time playlist updates."
    )

    st.markdown("#### Strategy 3: Hybrid (What most companies do)")
    st.graphviz_chart("""
        digraph strat3 {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            rt [label="Real-time\\nUser factor updates\\n(per purchase)", fillcolor="#d1fae5", color="#10b981"]
            nightly [label="Nightly\\nFull ALS retrain\\n(safety net)", fillcolor="#fef3c7", color="#f59e0b"]
            weekly [label="Weekly\\nFull pipeline rerun\\n(new items, new features)", fillcolor="#fce7f3", color="#ec4899"]

            rt -> nightly [label="  catches user drift", style=dashed]
            nightly -> weekly [label="  catches item drift", style=dashed]
        }
    """)
    st.markdown(
        "Real-time updates handle new purchases immediately. "
        "Nightly retrains catch new items and redistribute factors. "
        "Weekly pipeline reruns handle schema changes and new feature engineering."
    )

    st.markdown("---")
    st.subheader("Production Architecture for Recommendations")
    st.graphviz_chart("""
        digraph prod_rec {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            subgraph cluster_events {
                label="User Events"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"
                buy [label="Purchase", fillcolor="#dbeafe", color="#3b82f6"]
                view [label="Page View", fillcolor="#dbeafe", color="#3b82f6"]
                cart [label="Add to Cart", fillcolor="#dbeafe", color="#3b82f6"]
            }

            stream [label="Event Stream\\n(Kafka)", fillcolor="#e0e7ff", color="#6366f1"]
            matrix [label="Update Interaction\\nMatrix (real-time)", fillcolor="#fce7f3", color="#ec4899"]

            subgraph cluster_model {
                label="Model Layer"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"
                als [label="ALS Model\\n(user & item factors)", fillcolor="#d1fae5", color="#10b981"]
                update [label="Recompute user\\nfactors for affected\\ncustomers", fillcolor="#fef3c7", color="#f59e0b"]
            }

            api [label="Recommendation API\\nGET /recommend/user123\\n→ top 10 products", fillcolor="#d1fae5", color="#10b981"]

            subgraph cluster_serve {
                label="Serving"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"
                home [label="Homepage\\n'Recommended for you'", fillcolor="#fef3c7", color="#f59e0b"]
                email [label="Email\\nCampaigns", fillcolor="#fef3c7", color="#f59e0b"]
                pdp [label="Product Page\\n'Customers also bought'", fillcolor="#fef3c7", color="#f59e0b"]
            }

            buy -> stream
            view -> stream
            cart -> stream
            stream -> matrix -> update
            update -> als
            als -> api
            api -> home
            api -> email
            api -> pdp
        }
    """)

    st.markdown(
        """
        | Component | What it does | Technology |
        |---|---|---|
        | **Event Stream** | Captures every user action in real time | Kafka, Kinesis |
        | **Interaction Matrix** | Updates user-item counts as events arrive | Redis, DynamoDB |
        | **Factor Update** | Recomputes user factors for affected users | implicit library, custom ALS solver |
        | **ALS Model** | Stores user and item factor matrices | NumPy arrays in memory / Redis |
        | **Recommendation API** | Computes U[user] × Vᵀ, returns top-K items | FastAPI, gRPC |
        | **Serving Layer** | Integrates recommendations into the product | Website, email, mobile app |
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Live Demo
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("Streaming Simulation — Monthly Data Batches")
    st.markdown(
        """
        **What happens when you press Start:**
        1. We train an ALS model on the first **6 months** of purchase data (Dec 2010 – May 2011)
        2. We then stream in one month at a time (Jun – Nov 2011)
        3. After each month, we **retrain the model** on all data seen so far
        4. We evaluate how Hit Rate and Precision@10 improve as the model sees more data

        This simulates a production system that retrains nightly or weekly as new data accumulates.
        """
    )

    st.markdown("---")

    clean_df = load_clean_transactions()
    interactions_full, user_map, item_map, item_desc, _ = build_interaction_matrix()
    all_users = list(interactions_full.index)
    all_items = list(interactions_full.columns)
    n_users_total = len(all_users)
    n_items_total = len(all_items)

    clean_df["YearMonth"] = clean_df["InvoiceDate"].dt.to_period("M")
    months_sorted = sorted(clean_df["YearMonth"].unique())

    n_initial = st.slider("Initial training months", 3, 8, 6)
    n_factors = st.slider("Latent factors (k)", 20, 80, 50, step=10)
    delay = st.slider("Delay between months (seconds)", 0.5, 3.0, 1.0, step=0.5)

    initial_months = months_sorted[:n_initial]
    stream_months = months_sorted[n_initial:]

    st.markdown(
        f"**Initial training:** {initial_months[0]} – {initial_months[-1]} "
        f"({len(initial_months)} months)  \n"
        f"**Streaming batches:** {stream_months[0]} – {stream_months[-1]} "
        f"({len(stream_months)} months)"
    )

    # Build test set from the final month
    test_month = months_sorted[-1]
    test_transactions = clean_df[clean_df["YearMonth"] == test_month]
    test_user_item = (
        test_transactions.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )
    user_index = {u: i for i, u in enumerate(all_users)}
    item_index_map = {it: i for i, it in enumerate(all_items)}

    test_entries = []
    for _, row in test_user_item.iterrows():
        if row["CustomerID"] in user_index and row["StockCode"] in item_index_map:
            test_entries.append((row["CustomerID"], row["StockCode"], row["Quantity"]))
    test_df = pd.DataFrame(test_entries, columns=["CustomerID", "StockCode", "score"])

    st.markdown("---")

    if st.button("Start Streaming", type="primary", use_container_width=True):

        history = {"Hit Rate": [], "Precision@10": [], "Active Users": []}
        batch_labels = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_container = st.empty()
        metrics_row = st.empty()

        total_steps = len(stream_months)

        for step_i, month_cutoff_i in enumerate(range(n_initial, len(months_sorted))):
            months_used = months_sorted[:month_cutoff_i + 1]
            current_month = months_sorted[month_cutoff_i]
            batch_labels.append(str(current_month))

            cumulative_df = clean_df[clean_df["YearMonth"].isin(months_used)]
            train_sparse = build_matrix_from_transactions(cumulative_df, all_users, all_items)

            n_active = (train_sparse.sum(axis=1) > 0).sum()
            n_interactions = train_sparse.nnz

            status_text.markdown(
                f"**{current_month}:** Training ALS on {len(months_used)} months "
                f"({n_interactions:,} interactions, {int(n_active):,} active users)..."
            )

            model = AlternatingLeastSquares(
                factors=n_factors, iterations=15, regularization=0.1, random_state=42,
            )
            model.fit(train_sparse)

            predicted = model.user_factors @ model.item_factors.T
            train_dense = train_sparse.toarray()

            hits = 0
            precisions = []
            n_eval = 0
            for cust_id, group in test_df.groupby("CustomerID"):
                if cust_id not in user_index:
                    continue
                uidx = user_index[cust_id]
                true_items = set()
                for _, r in group.iterrows():
                    if r["StockCode"] in item_index_map:
                        true_items.add(item_index_map[r["StockCode"]])
                if not true_items:
                    continue

                scores = predicted[uidx].copy()
                scores[train_dense[uidx] > 0] = -np.inf
                top10 = np.argsort(scores)[::-1][:10]
                hit_count = len(set(top10) & true_items)
                precisions.append(hit_count / 10)
                if hit_count > 0:
                    hits += 1
                n_eval += 1

            hr = hits / n_eval if n_eval > 0 else 0
            p10 = float(np.mean(precisions)) if precisions else 0

            history["Hit Rate"].append(hr)
            history["Precision@10"].append(p10)
            history["Active Users"].append(int(n_active))

            progress_bar.progress((step_i + 1) / total_steps)

            with metrics_row.container():
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Hit Rate",
                    f"{hr:.1%}",
                    f"{hr - history['Hit Rate'][-2]:+.1%}" if len(history["Hit Rate"]) > 1 else None,
                )
                m2.metric(
                    "Precision@10",
                    f"{p10:.4f}",
                    f"{p10 - history['Precision@10'][-2]:+.4f}" if len(history["Precision@10"]) > 1 else None,
                )
                m3.metric("Months of Data", f"{len(months_used)}")

            with chart_container.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=batch_labels, y=history["Hit Rate"],
                    mode="lines+markers", name="Hit Rate",
                    line=dict(color="#636EFA", width=3),
                ))
                fig.add_trace(go.Scatter(
                    x=batch_labels, y=history["Precision@10"],
                    mode="lines+markers", name="Precision@10",
                    line=dict(color="#EF553B", width=3),
                    yaxis="y2",
                ))
                fig.update_layout(
                    title="Recommendation Quality Over Time",
                    xaxis_title="Month Added",
                    yaxis=dict(title="Hit Rate", side="left", tickformat=".0%"),
                    yaxis2=dict(title="Precision@10", side="right", overlaying="y", tickformat=".4f"),
                    legend=dict(x=0.01, y=0.99),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"rec_live_{step_i}")

            time.sleep(delay)

        progress_bar.progress(1.0)
        status_text.success(
            f"Streaming complete! Model evolved from {n_initial} months to {len(months_sorted)} months of data."
        )

        st.markdown("---")
        st.subheader("Summary")

        initial_hr = history["Hit Rate"][0]
        final_hr = history["Hit Rate"][-1]
        initial_p = history["Precision@10"][0]
        final_p = history["Precision@10"][-1]

        st.markdown(
            f"""
            | Metric | After {n_initial} Months | After {len(months_sorted)} Months | Improvement |
            |---|---|---|---|
            | Hit Rate | {initial_hr:.1%} | {final_hr:.1%} | {final_hr - initial_hr:+.1%} |
            | Precision@10 | {initial_p:.4f} | {final_p:.4f} | {final_p - initial_p:+.4f} |
            """
        )

        st.markdown(
            """
            **What this demonstrates:** As the model sees more purchase history, it learns
            better patterns and makes more accurate recommendations. Each month of new data
            adds signal — especially for new customers who didn't exist in early months
            and for seasonal patterns that only become visible over time.

            In a production system, this would run automatically: nightly batch retrains
            using all accumulated data, supplemented by real-time user factor updates
            for immediate responsiveness.
            """
        )
    else:
        st.info("Click **Start Streaming** to watch the recommendation model improve as monthly data arrives.")
