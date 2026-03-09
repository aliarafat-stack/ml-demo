import streamlit as st
from utils.data_loader import load_raw_data
from utils.rec_data_loader import load_clean_transactions, build_interaction_matrix

st.set_page_config(
    page_title="Data Science Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Data Science Project Demo")
st.markdown(
    "An end-to-end demonstration of two core data science objectives: **predicting customer churn** "
    "and **building a product recommendation system** — from raw data to deployed models."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# KPI Dashboard — Both Objectives
# ═══════════════════════════════════════════════════════════════════════════════
churn_col, rec_col = st.columns(2)

with churn_col:
    st.markdown("### Customer Churn Prediction")
    df_churn = load_raw_data()
    total_customers = len(df_churn)
    churn_rate = df_churn["Churn"].mean()
    churned_count = int(df_churn["Churn"].sum())
    revenue_at_risk = df_churn.loc[df_churn["Churn"] == 1, "MonthlyCharges"].sum() * 12

    c1, c2 = st.columns(2)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churned_count:,} lost", delta_color="inverse")

    c3, c4 = st.columns(2)
    c3.metric("Avg Monthly Charge", f"${df_churn['MonthlyCharges'].mean():,.2f}")
    c4.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.caption("Dataset: Telco Customer Churn — 7,043 customers, 21 features")

with rec_col:
    st.markdown("### Product Recommendation")
    df_rec = load_clean_transactions()
    interactions, *_ = build_interaction_matrix()
    n_users, n_items = interactions.shape
    n_transactions = len(df_rec)
    sparsity = 1 - ((interactions > 0).sum().sum() / (n_users * n_items))
    total_revenue = df_rec["TotalPrice"].sum()

    r1, r2 = st.columns(2)
    r1.metric("Unique Customers", f"{n_users:,}")
    r2.metric("Unique Products", f"{n_items:,}")

    r3, r4 = st.columns(2)
    r3.metric("Transactions", f"{n_transactions:,}")
    r4.metric("Matrix Sparsity", f"{sparsity:.1%}")

    st.caption("Dataset: UCI Online Retail — 541K transactions, UK e-commerce 2010–2011")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Two Objectives Overview
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Two Objectives — Full Pipeline for Each")

st.markdown(
    """
    | Objective | Problem Type | Key Algorithms | Pages |
    |---|---|---|---|
    | **Customer Churn** | Binary Classification | Logistic Regression, Random Forest, XGBoost, SGDClassifier | EDA → Preprocessing → Churn Models → Live Updates |
    | **Product Recommendation** | Matrix Factorization | SVD, ALS, NMF, Item-Based CF | Rec EDA → Rec Preprocessing → Rec Models → Rec Live |
    """
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# The ML Journey — Step-by-step pipeline
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("The Machine Learning Journey — What We're Doing and Why")
st.markdown(
    "Both objectives follow the same well-defined pipeline. "
    "Each step has a purpose and builds on the previous one:"
)

st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.3,0.15"]
        edge [color="#888888", penwidth=1.5]

        data    [label="1. Data\\nCollection",   fillcolor="#dbeafe", color="#3b82f6"]
        eda     [label="2. Exploratory\\nData Analysis", fillcolor="#e0e7ff", color="#6366f1"]
        preproc [label="3. Data\\nPreprocessing",  fillcolor="#fce7f3", color="#ec4899"]
        train   [label="4. Model\\nTraining",      fillcolor="#d1fae5", color="#10b981"]
        eval    [label="5. Evaluation\\n& Comparison", fillcolor="#fef3c7", color="#f59e0b"]
        deploy  [label="6. Live Updates\\n& Deployment", fillcolor="#fee2e2", color="#ef4444"]

        data -> eda -> preproc -> train -> eval -> deploy
    }
""")

steps = [
    {
        "num": "1",
        "title": "Data Collection",
        "churn": "**Telco Customer Churn** — 7,043 customers with demographics, services, billing, and churn labels.",
        "rec": "**UCI Online Retail** — 541K purchase transactions from a UK e-commerce store.",
        "why": "You can't build a model without data. The quality and relevance of the data determines everything that follows.",
    },
    {
        "num": "2",
        "title": "Exploratory Data Analysis",
        "churn": "Visualize churn distribution, feature histograms, category-based churn rates, correlation heatmaps, and UMAP customer segments.",
        "rec": "Analyze purchase distributions, product popularity (long tail), user-item matrix sparsity, revenue trends, and geography.",
        "why": "Before building models, we need to understand the problem. EDA reveals patterns, catches data issues, and guides modeling choices.",
    },
    {
        "num": "3",
        "title": "Data Preprocessing",
        "churn": "Handle missing values (median imputation), encode categories (Label Encoding for trees, One-Hot for Logistic Regression), scale features (StandardScaler).",
        "rec": "Clean transactions (remove cancellations, missing IDs), aggregate into a user-item matrix, handle implicit feedback.",
        "why": "ML algorithms need clean, structured input. Raw data has missing values, text, and different scales — all of which confuse models.",
    },
    {
        "num": "4",
        "title": "Model Training",
        "churn": "Train **Logistic Regression**, **Random Forest**, and **XGBoost** — each with the encoding that's optimal for it.",
        "rec": "Train **SVD**, **ALS**, **NMF**, and **Item-Based CF** — four approaches to matrix factorization and collaborative filtering.",
        "why": "Different algorithms have different strengths. Training multiple models lets us find the best approach for each problem.",
    },
    {
        "num": "5",
        "title": "Evaluation & Comparison",
        "churn": "Compare using Accuracy, Precision, Recall, F1, AUC. Explain predictions with **SHAP**.",
        "rec": "Compare using Precision@K, Recall@K, Hit Rate. Inspect individual recommendations for quality.",
        "why": "A model is only useful if it's accurate and we can measure how accurate. Different metrics capture different aspects of quality.",
    },
    {
        "num": "6",
        "title": "Live Updates & Deployment",
        "churn": "Incremental learning with **SGDClassifier.partial_fit()** — update the model as new customer data streams in.",
        "rec": "Monthly batch retraining of **ALS** — model quality improves as purchase history accumulates over time.",
        "why": "In the real world, data doesn't stop arriving. Deployed models must adapt to new patterns without expensive full retraining.",
    },
]

for step in steps:
    with st.expander(f"Step {step['num']}: {step['title']}", expanded=False):
        st.markdown(f"**Why:** {step['why']}")
        col_churn, col_rec = st.columns(2)
        with col_churn:
            st.markdown(f"**Churn Prediction:**  \n{step['churn']}")
        with col_rec:
            st.markdown(f"**Recommendation:**  \n{step['rec']}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# How a model actually "learns" — quick primer
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("How Does a Machine Learning Model Actually \"Learn\"?")
st.markdown(
    """
    At its core, every ML model follows the same loop:
    """
)

st.graphviz_chart("""
    digraph learning {
        rankdir=LR
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.3,0.15"]
        edge [color="#888888", penwidth=1.5]

        input  [label="Input Data\\n(features)", fillcolor="#dbeafe", color="#3b82f6"]
        guess  [label="Model Makes\\na Prediction",  fillcolor="#d1fae5", color="#10b981"]
        check  [label="Compare to\\nActual Answer",  fillcolor="#fef3c7", color="#f59e0b"]
        adjust [label="Adjust Model\\n(reduce error)", fillcolor="#fee2e2", color="#ef4444"]

        input -> guess -> check -> adjust
        adjust -> guess [style=dashed, label="repeat", fontsize=9]
    }
""")

st.markdown(
    """
    1. **Input** — The model receives data (customer features for churn, purchase history for recommendations)
    2. **Predict** — It makes a guess (will they churn? what would they buy?)
    3. **Compare** — We check the guess against reality
    4. **Adjust** — The model tweaks its internal parameters to reduce the error
    5. **Repeat** — This cycle runs until the model gets good

    Different algorithms differ in *how* they make predictions and *how* they adjust.
    The algorithm explanation pages cover each one with diagrams.
    """
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Business Impact
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Business Impact")

impact_churn, impact_rec = st.columns(2)
with impact_churn:
    st.markdown("#### Churn Reduction")
    reduction_pct = st.slider(
        "If we reduce churn by this %",
        min_value=1, max_value=50, value=10, step=1, format="%d%%",
    )
    savings = revenue_at_risk * (reduction_pct / 100)
    st.success(f"**{reduction_pct}% churn reduction** → **${savings:,.0f}/year** saved")

with impact_rec:
    st.markdown("#### Recommendation Revenue")
    avg_order = df_rec["TotalPrice"].mean()
    conversion_pct = st.slider(
        "If recommendations lift conversion by this %",
        min_value=1, max_value=30, value=5, step=1, format="%d%%",
    )
    additional = total_revenue * (conversion_pct / 100)
    st.success(f"**{conversion_pct}% conversion lift** → **£{additional:,.0f}/year** additional revenue")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Page Directory
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Page Directory")

st.markdown(
    """
    **Customer Churn Prediction:**

    | Page | What You'll See |
    |---|---|
    | **EDA** | Interactive charts exploring customer demographics, services, and churn patterns |
    | **Preprocessing** | Step-by-step data cleaning, encoding (Label + One-Hot), scaling with before/after visuals |
    | **Churn Models** | Algorithm explainers with diagrams, model comparison, SHAP explanations, What-If predictor |
    | **Live Updates** | Real-time incremental learning demo with SGDClassifier partial_fit |

    **Product Recommendation:**

    | Page | What You'll See |
    |---|---|
    | **Rec EDA** | Purchase distributions, product popularity (long tail), sparsity analysis, user-item matrix |
    | **Rec Preprocessing** | Transaction-to-matrix pipeline, implicit vs explicit feedback, train/test split |
    | **Rec Models** | Algorithm explainers for SVD/ALS/NMF/Item-CF, model comparison, interactive recommendations |
    | **Rec Live** | Monthly data streaming demo — watch ALS improve as purchase history grows |
    """
)

st.markdown("---")
st.caption(
    "Data: Telco Customer Churn Dataset + UCI Online Retail Dataset • "
    "Built with Streamlit, scikit-learn, XGBoost, SHAP, implicit, scipy"
)
