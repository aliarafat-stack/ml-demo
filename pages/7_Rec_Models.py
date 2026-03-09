import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.rec_data_loader import build_interaction_matrix, get_rec_train_test
from utils.rec_models import (
    train_svd, train_als, train_nmf, train_item_cf,
    get_top_n_recommendations, evaluate_recommendations,
)

st.set_page_config(page_title="Rec Models", page_icon="🎯", layout="wide")
st.title("Product Recommendation — Models")
st.markdown("---")

tab_how, tab_compare, tab_demo = st.tabs([
    "How The Algorithms Work", "Model Comparison", "Interactive Recommendations"
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — How The Algorithms Work
# ═══════════════════════════════════════════════════════════════════════════════
with tab_how:
    st.subheader("Recommendation Algorithms — A Visual Guide")
    st.markdown(
        "All these algorithms solve the same problem: **fill in the empty cells** "
        "of the user-item matrix. They just approach it differently."
    )

    # ── Matrix Factorization overview ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### The Core Idea: Matrix Factorization")
    st.markdown(
        "Matrix factorization is like discovering the **hidden reasons** behind purchases."
    )

    st.graphviz_chart("""
        digraph mf {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            R [label="User-Item Matrix\\nR (4338 × 3665)\\nMostly empty", fillcolor="#dbeafe", color="#3b82f6"]
            approx [label="≈", shape=plaintext, fontsize=24]
            U [label="User Matrix\\nU (4338 × k)\\nEach user as k\\nlatent preferences", fillcolor="#d1fae5", color="#10b981"]
            times [label="×", shape=plaintext, fontsize=24]
            V [label="Item Matrix\\nV (k × 3665)\\nEach item as k\\nlatent attributes", fillcolor="#fce7f3", color="#ec4899"]

            R -> approx [arrowhead=none]
            approx -> U [arrowhead=none]
            U -> times [arrowhead=none]
            times -> V [arrowhead=none]
        }
    """)

    st.markdown(
        """
        **The intuition with an example:**

        Suppose there are 3 hidden factors: "kitchen items", "vintage style", "party supplies".
        - **User profile** (U): Customer 12345 = [0.9 kitchen, 0.1 vintage, 0.7 party]
        - **Item profile** (V): "Cake Stand" = [0.8 kitchen, 0.3 vintage, 0.5 party]
        - **Predicted score** = 0.9×0.8 + 0.1×0.3 + 0.7×0.5 = **1.10** → strong recommendation!

        The model **discovers** these latent factors automatically from the purchase patterns.
        We set the number of factors (k), and the algorithm figures out what they represent.
        """
    )

    # ── SVD ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. SVD (Singular Value Decomposition)")
    st.markdown("*The classic matrix factorization technique from linear algebra.*")

    st.graphviz_chart("""
        digraph svd {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            R [label="R\\nUser-Item\\nMatrix", fillcolor="#dbeafe", color="#3b82f6"]
            eq [label="=", shape=plaintext, fontsize=24]
            U [label="U\\nUser factors\\n(users × k)", fillcolor="#d1fae5", color="#10b981"]
            S [label="Σ\\nSingular values\\n(k × k diagonal)\\nimportance weights", fillcolor="#fef3c7", color="#f59e0b"]
            Vt [label="Vᵀ\\nItem factors\\n(k × items)", fillcolor="#fce7f3", color="#ec4899"]

            R -> eq [arrowhead=none]
            eq -> U [arrowhead=none]
            U -> S [label=" × "]
            S -> Vt [label=" × "]
        }
    """)

    st.markdown(
        """
        **How it works:**
        1. SVD decomposes the user-item matrix R into three matrices: **U × Σ × Vᵀ**
        2. **U** captures user preferences in k-dimensional "latent space"
        3. **Σ** (Sigma) is a diagonal matrix of importance weights — the first few factors
           capture the most important patterns, the rest are noise
        4. **Vᵀ** captures item characteristics in the same latent space
        5. We keep only the top **k** factors (truncated SVD) — this filters noise and
           fills in the missing values with predictions

        **Strengths:** Mathematically elegant, well-understood, fast with sparse matrices.
        **Weaknesses:** Treats missing entries as zero (problematic for implicit feedback —
        "didn't buy" ≠ "doesn't like").
        """
    )

    # ── ALS ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. ALS (Alternating Least Squares)")
    st.markdown("*The industry standard for implicit feedback — used at Spotify and Netflix.*")

    st.graphviz_chart("""
        digraph als {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            init [label="Initialize\\nRandom user & item\\nfactor matrices", fillcolor="#dbeafe", color="#3b82f6"]
            fix_u [label="Fix User factors\\nOptimize Item factors\\n(least squares)", fillcolor="#d1fae5", color="#10b981"]
            fix_i [label="Fix Item factors\\nOptimize User factors\\n(least squares)", fillcolor="#fce7f3", color="#ec4899"]
            check [label="Check convergence\\n(error decreasing?)", fillcolor="#fef3c7", color="#f59e0b"]
            done [label="Final U and V\\nPredict: U × Vᵀ", fillcolor="#e0e7ff", color="#6366f1"]

            init -> fix_u -> fix_i -> check
            check -> fix_u [label="not converged", style=dashed]
            check -> done [label="converged"]
        }
    """)

    st.markdown(
        """
        **How it works:**
        1. Start with random user factors (U) and item factors (V)
        2. **Fix U**, solve for the best V (this is a standard least squares problem)
        3. **Fix V**, solve for the best U (again, least squares)
        4. **Alternate** between steps 2 and 3 until the error stops decreasing
        5. Multiply U × Vᵀ to get predicted scores for every user-item pair

        **Why "Alternating"?** Optimizing U and V simultaneously is hard (non-convex).
        But if you fix one, optimizing the other becomes a simple, solvable least-squares
        problem. By alternating, we make steady progress toward a good solution.

        **Key advantage for implicit feedback:** ALS treats observed interactions (purchases)
        and unobserved ones (empty cells) differently. It assigns **confidence weights** —
        a purchase of 10 units gets higher confidence than 1 unit, while empty cells get
        low (but non-zero) confidence. This is exactly right for our e-commerce data.

        **Strengths:** Designed for implicit data, parallelizable, scales to millions of users.
        **Weaknesses:** More complex to implement, requires tuning regularization.
        """
    )

    # ── NMF ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. NMF (Non-negative Matrix Factorization)")
    st.markdown("*Matrix factorization with a twist: everything stays positive.*")

    st.graphviz_chart("""
        digraph nmf {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            R [label="R\\nUser-Item Matrix\\n(all values ≥ 0)", fillcolor="#dbeafe", color="#3b82f6"]
            approx [label="≈", shape=plaintext, fontsize=24]
            W [label="W\\nUser factors\\n(all values ≥ 0)\\nadditive parts only", fillcolor="#d1fae5", color="#10b981"]
            times [label="×", shape=plaintext, fontsize=24]
            H [label="H\\nItem factors\\n(all values ≥ 0)\\nadditive parts only", fillcolor="#fce7f3", color="#ec4899"]

            R -> approx [arrowhead=none]
            approx -> W [arrowhead=none]
            W -> times [arrowhead=none]
            times -> H [arrowhead=none]
        }
    """)

    st.markdown(
        """
        **How it's different from SVD:**
        - SVD factors can be negative (hard to interpret: what does a "-0.5 kitchen" preference mean?)
        - NMF forces **all values to be ≥ 0** — factors are purely additive

        **Why non-negative is useful:**
        - A user is "0.8 kitchen + 0.3 vintage + 0 party" — easy to interpret
        - Each factor represents an additive "part" of the user's taste
        - The results are **naturally sparse** — many factors are exactly zero
        - This makes NMF factors more interpretable as "topics" or "themes"

        **Analogy:** If SVD says "you like kitchen items but with a negative party component",
        NMF says "you're 80% kitchen enthusiast, 30% vintage lover, and don't care about party
        supplies." The NMF interpretation is more natural for recommendation.

        **Strengths:** Interpretable factors, naturally sparse, good for parts-based decomposition.
        **Weaknesses:** Only works with non-negative data, sensitive to initialization.
        """
    )

    # ── Item-Based CF ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Item-Based Collaborative Filtering")
    st.markdown("*No factorization — just find similar items.*")

    st.graphviz_chart("""
        digraph itemcf {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            bought [label="Customer bought\\nCake Stand", fillcolor="#dbeafe", color="#3b82f6"]
            sim [label="Find items with\\nsimilar purchase patterns\\n(cosine similarity)", fillcolor="#e0e7ff", color="#6366f1"]
            items [label="Similar items:\\n1. Cake Tins (0.85)\\n2. Baking Set (0.79)\\n3. Cookie Cutters (0.72)", fillcolor="#d1fae5", color="#10b981"]
            rec [label="Recommend\\nthese items!", fillcolor="#fef3c7", color="#f59e0b"]

            bought -> sim -> items -> rec
        }
    """)

    st.markdown(
        """
        **How it works:**
        1. For every pair of items, compute **cosine similarity** based on their purchase patterns
           (items bought by similar sets of customers are similar)
        2. When recommending for a user, look at items they've already bought
        3. For each purchased item, find its most similar items
        4. Rank by similarity score and recommend the top ones (excluding already-purchased)

        **Why cosine similarity?** It measures the angle between two vectors, ignoring magnitude.
        Two items with purchase patterns [1, 0, 1, 1] and [3, 0, 2, 4] have high cosine similarity
        (same direction) even though the quantities differ.

        **Strengths:** Simple, intuitive, no training needed, easy to explain.
        **Weaknesses:** Doesn't discover latent factors, doesn't work well with very sparse data,
        O(n²) item-pair computation.
        """
    )

    # ── Comparison Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Quick Comparison")
    st.markdown(
        """
        | | SVD | ALS | NMF | Item-Based CF |
        |---|---|---|---|---|
        | **Approach** | Decompose R = UΣVᵀ | Alternate optimizing U and V | Decompose R ≈ W×H (non-negative) | Find similar items via cosine |
        | **Feedback type** | Explicit (treats 0 as 0) | Implicit (0 = unknown, not dislike) | Non-negative values | Either |
        | **Interpretability** | Low (negative factors) | Low | High (additive parts) | Very high (similar items) |
        | **Scalability** | Good | Excellent (parallelizable) | Moderate | Poor for many items |
        | **Used by** | Research, baselines | Spotify, Netflix, LinkedIn | Topic modeling, parts decomposition | Amazon ("customers also bought") |
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Model Performance Comparison")
    st.markdown(
        "We train all four algorithms on the same training set and evaluate "
        "on held-out test interactions using Precision@10, Recall@10, and Hit Rate."
    )

    interactions, user_map, item_map, item_desc, _ = build_interaction_matrix()
    train_df, train_sparse, test_df, _, _, _ = get_rec_train_test()
    train_dense = train_df.values.astype(np.float32)

    user_index = {cust: i for i, cust in enumerate(train_df.index)}
    item_columns = list(train_df.columns)

    n_factors = st.slider("Number of latent factors (k)", 10, 100, 50, step=10)

    if st.button("Train & Evaluate All Models", type="primary", use_container_width=True):
        results = {}

        with st.spinner("Training SVD..."):
            svd_pred, *_ = train_svd(train_sparse, n_factors=n_factors)
            results["SVD"] = evaluate_recommendations(svd_pred, train_sparse, test_df, user_index, item_columns)
            st.success("SVD trained")

        with st.spinner("Training ALS..."):
            als_model = train_als(train_sparse, n_factors=n_factors)
            als_user = als_model.user_factors
            als_item = als_model.item_factors
            als_pred = als_user @ als_item.T
            results["ALS"] = evaluate_recommendations(als_pred, train_sparse, test_df, user_index, item_columns)
            st.success("ALS trained")

        with st.spinner("Training NMF..."):
            nmf_pred, *_ = train_nmf(train_dense, n_factors=n_factors)
            results["NMF"] = evaluate_recommendations(nmf_pred, train_sparse, test_df, user_index, item_columns)
            st.success("NMF trained")

        with st.spinner("Training Item-Based CF..."):
            cf_pred, _ = train_item_cf(train_sparse)
            results["Item-CF"] = evaluate_recommendations(cf_pred, train_sparse, test_df, user_index, item_columns)
            st.success("Item-Based CF trained")

        st.markdown("---")
        st.markdown("### Results")

        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df[["Precision@K", "Recall@K", "Hit Rate", "Users Evaluated"]]
        metrics_df["Precision@K"] = metrics_df["Precision@K"].map("{:.4f}".format)
        metrics_df["Recall@K"] = metrics_df["Recall@K"].map("{:.4f}".format)
        metrics_df["Hit Rate"] = metrics_df["Hit Rate"].map("{:.2%}".format)
        metrics_df["Users Evaluated"] = metrics_df["Users Evaluated"].astype(int)

        st.dataframe(metrics_df, use_container_width=True)

        raw_results = pd.DataFrame(results).T
        fig_compare = go.Figure()
        for metric in ["Precision@K", "Recall@K", "Hit Rate"]:
            fig_compare.add_trace(go.Bar(
                name=metric,
                x=list(results.keys()),
                y=raw_results[metric].values,
                text=[f"{v:.4f}" for v in raw_results[metric].values],
                textposition="auto",
            ))
        fig_compare.update_layout(
            title="Model Comparison",
            barmode="group",
            yaxis_title="Score",
            height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown(
            """
            **How to read these results:**
            - **Precision@10:** Higher = fewer irrelevant recommendations in the top 10
            - **Recall@10:** Higher = more of the user's actual purchases found in top 10
            - **Hit Rate:** Higher = more users got at least one relevant recommendation

            In e-commerce, even low absolute numbers (e.g., Precision@10 = 0.05 means 0.5 out
            of 10 recommendations is relevant) can be valuable — the user-item space is enormous
            and finding *any* relevant niche item is useful.
            """
        )

        st.session_state["rec_models_trained"] = True
        st.session_state["svd_pred"] = svd_pred
        st.session_state["als_pred"] = als_pred
        st.session_state["nmf_pred"] = nmf_pred
        st.session_state["cf_pred"] = cf_pred
        st.session_state["train_df"] = train_df
        st.session_state["item_desc"] = item_desc
        st.session_state["item_map"] = item_map
        st.session_state["user_map"] = user_map
    else:
        st.info("Click the button above to train all models. This may take 30–60 seconds on first run.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Interactive Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("Interactive Recommendation Demo")
    st.markdown(
        "Select a customer and see what each model recommends. "
        "You can also see what the customer has actually purchased to judge quality."
    )

    if not st.session_state.get("rec_models_trained"):
        st.warning("Please train the models first in the **Model Comparison** tab.")
    else:
        train_df = st.session_state["train_df"]
        item_desc = st.session_state["item_desc"]
        item_map = st.session_state["item_map"]

        customer_list = list(train_df.index)
        active_customers = [c for c in customer_list if (train_df.loc[c] > 0).sum() >= 10]

        selected_customer = st.selectbox(
            "Select a Customer",
            active_customers[:100],
            format_func=lambda x: f"Customer {int(x)} ({int((train_df.loc[x] > 0).sum())} products purchased)",
        )

        user_idx = customer_list.index(selected_customer)

        st.markdown("#### What this customer has purchased")
        purchased = train_df.loc[selected_customer]
        purchased_items = purchased[purchased > 0].sort_values(ascending=False).head(15)
        purch_display = pd.DataFrame({
            "StockCode": purchased_items.index,
            "Quantity": purchased_items.values.astype(int),
            "Description": [item_desc.get(sc, "N/A") for sc in purchased_items.index],
        })
        st.dataframe(purch_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Recommendations from each model")

        model_preds = {
            "SVD": st.session_state["svd_pred"],
            "ALS": st.session_state["als_pred"],
            "NMF": st.session_state["nmf_pred"],
            "Item-CF": st.session_state["cf_pred"],
        }

        n_recs = st.slider("Number of recommendations", 5, 20, 10)

        cols = st.columns(2)
        for i, (model_name, pred_matrix) in enumerate(model_preds.items()):
            with cols[i % 2]:
                st.markdown(f"**{model_name}**")
                top_items, top_scores = get_top_n_recommendations(
                    pred_matrix, train_df.values, user_idx, n=n_recs,
                )
                rec_display = pd.DataFrame({
                    "Rank": range(1, len(top_items) + 1),
                    "StockCode": [train_df.columns[j] for j in top_items],
                    "Score": [f"{s:.3f}" for s in top_scores],
                    "Description": [item_desc.get(train_df.columns[j], "N/A") for j in top_items],
                })
                st.dataframe(rec_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            "**How to evaluate these recommendations:** Look at the customer's purchase "
            "history above and check if the recommended items make sense. For example, if "
            "the customer bought a lot of kitchen items, good recommendations would include "
            "other kitchen products they haven't purchased yet."
        )
