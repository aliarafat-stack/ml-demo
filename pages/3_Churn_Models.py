import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from utils.data_loader import (
    load_raw_data, get_train_test, get_onehot_train_test,
    get_encoded_data, CATEGORICAL_COLS, NUMERIC_COLS,
)
from utils.models import (
    train_tree_models, train_lr_model, evaluate_model,
    get_shap_explainer, get_shap_single,
)
from utils.visualizations import plot_roc_curves, plot_confusion_matrix, plot_gauge

st.set_page_config(page_title="Churn Models", page_icon="🤖", layout="wide")
st.title("Churn Prediction Models")
st.markdown("---")

X_train, X_test, y_train, y_test, encoders, feature_cols = get_train_test()
X_train_oh, X_test_oh, _, _, feature_cols_oh = get_onehot_train_test()

with st.spinner("Training models (cached after first run)..."):
    tree_models = train_tree_models(X_train, y_train)
    lr_model = train_lr_model(X_train_oh, y_train)

# Unified structures for iteration:
#   all_models: ordered dict of name -> model
#   model_test_data: name -> X_test appropriate for that model
all_models = {"Logistic Regression": lr_model}
all_models.update(tree_models)

model_test_data = {
    "Logistic Regression": X_test_oh,
    "Random Forest": X_test,
    "XGBoost": X_test,
}

metrics = {}
for name, model in all_models.items():
    metrics[name] = evaluate_model(model, model_test_data[name], y_test)

st.info(
    "**Encoding note:** Logistic Regression is trained on **One-Hot Encoded** features "
    f"({len(feature_cols_oh)} columns) while Random Forest and XGBoost use "
    f"**Label Encoding** ({len(feature_cols)} columns). "
    "Each model gets the encoding that's optimal for it."
)

tab_how, tab_predict, tab_compare, tab_shap_global, tab_shap_individual, tab_whatif = st.tabs(
    ["How The Algorithms Work", "Predict and Compare", "Model Comparison", "Feature Importance (SHAP)", "Individual Explanations", "What-If Predictor"]
)

# ── Tab: How The Algorithms Work ─────────────────────────────────────────────
with tab_how:
    st.subheader("How Each Algorithm Works — A Visual Guide")
    st.markdown(
        "We use three very different algorithms to predict churn. Each approaches "
        "the problem in its own way. Understanding the differences helps explain "
        "why one model might outperform another."
    )

    # ── Logistic Regression ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Logistic Regression")
    st.markdown("*The simplest model — and often a strong baseline.*")

    st.graphviz_chart("""
        digraph lr {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.3,0.15"]
            edge [color="#888888", penwidth=1.5]

            features [label="Customer Features\\ntenure, charges,\\ncontract type, ...", fillcolor="#dbeafe", color="#3b82f6"]
            weighted [label="Weighted Sum\\nw₁×tenure + w₂×charges\\n+ w₃×contract + ...", fillcolor="#e0e7ff", color="#6366f1"]
            sigmoid  [label="Sigmoid Function\\nSquash to 0-1", fillcolor="#fce7f3", color="#ec4899"]
            output   [label="Probability\\n0.82 → Churn\\n0.15 → Retain", fillcolor="#d1fae5", color="#10b981"]

            features -> weighted -> sigmoid -> output
        }
    """)

    import plotly.graph_objects as go

    x_sig = np.linspace(-8, 8, 200)
    y_sig = 1 / (1 + np.exp(-x_sig))
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=x_sig, y=y_sig, mode="lines", line=dict(color="#6366f1", width=3), name="Sigmoid"))
    fig_sig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Decision boundary (0.5)")
    fig_sig.add_vrect(x0=-8, x1=0, fillcolor="#d1fae5", opacity=0.15, annotation_text="Retain", annotation_position="bottom left")
    fig_sig.add_vrect(x0=0, x1=8, fillcolor="#fee2e2", opacity=0.15, annotation_text="Churn", annotation_position="bottom right")
    fig_sig.update_layout(
        title="The Sigmoid Curve — Turns Any Number into a Probability",
        xaxis_title="Weighted Sum of Features (z = w₁x₁ + w₂x₂ + ...)",
        yaxis_title="Churn Probability",
        height=350,
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown(
        """
        **How it works in plain English:**
        1. Each customer feature gets a **weight** (a number the model learns). For example,
           "month-to-month contract" might get a high positive weight (pushes toward churn),
           while "long tenure" gets a negative weight (pushes away from churn).
        2. The model adds up: `weight₁ × feature₁ + weight₂ × feature₂ + ...`
        3. This sum could be any number (-∞ to +∞). The **sigmoid function** (the S-curve above)
           squashes it into a probability between 0 and 1.
        4. If the probability > 0.5 → predict "Churn". Otherwise → "Retain".

        **Why we use One-Hot Encoding for this model:** Because the weighted sum treats numbers
        at face value. If we encoded "Month-to-month"=2 and "Two year"=0, the model would
        think month-to-month is "more" of something than two-year — which is nonsensical.
        One-Hot avoids this by giving each category its own yes/no column.

        **Strengths:** Fast, interpretable (weights directly tell you what matters), good baseline.
        **Weaknesses:** Assumes a linear relationship between features and the log-odds of churn.
        Can't capture complex interactions (e.g., "fiber optic is only risky for short-tenure customers").
        """
    )

    # ── Random Forest ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. Random Forest")
    st.markdown("*An ensemble of decision trees that vote together.*")

    st.graphviz_chart("""
        digraph rf {
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            data [label="Training Data", fillcolor="#dbeafe", color="#3b82f6"]

            subgraph cluster_trees {
                label="200 Decision Trees (each sees a random subset)"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"
                fontsize=10

                t1 [label="Tree 1\\n→ Churn", fillcolor="#d1fae5", color="#10b981"]
                t2 [label="Tree 2\\n→ Retain", fillcolor="#d1fae5", color="#10b981"]
                t3 [label="Tree 3\\n→ Churn", fillcolor="#d1fae5", color="#10b981"]
                dots [label="...\\n(197 more)", shape=plaintext]
                t200 [label="Tree 200\\n→ Churn", fillcolor="#d1fae5", color="#10b981"]
            }

            vote [label="Majority Vote\\n3 out of 4 say Churn\\n→ Final: Churn (75%)", fillcolor="#fef3c7", color="#f59e0b"]

            data -> t1
            data -> t2
            data -> t3
            data -> t200
            t1 -> vote
            t2 -> vote
            t3 -> vote
            t200 -> vote
        }
    """)

    st.markdown("**How a single decision tree works:**")
    st.graphviz_chart("""
        digraph tree_example {
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.2,0.1"]
            edge [fontname="Helvetica", fontsize=9, color="#888888", penwidth=1.5]

            q1 [label="Contract = \\nMonth-to-month?", fillcolor="#e0e7ff", color="#6366f1"]
            q2 [label="tenure < 12\\nmonths?", fillcolor="#e0e7ff", color="#6366f1"]
            q3 [label="MonthlyCharges\\n> $70?", fillcolor="#e0e7ff", color="#6366f1"]
            churn1 [label="CHURN\\n(85% probability)", fillcolor="#fee2e2", color="#ef4444"]
            retain1 [label="RETAIN\\n(70% probability)", fillcolor="#d1fae5", color="#10b981"]
            churn2 [label="CHURN\\n(60% probability)", fillcolor="#fee2e2", color="#ef4444"]
            retain2 [label="RETAIN\\n(90% probability)", fillcolor="#d1fae5", color="#10b981"]

            q1 -> q2 [label=" Yes"]
            q1 -> q3 [label=" No"]
            q2 -> churn1 [label=" Yes"]
            q2 -> retain1 [label=" No"]
            q3 -> churn2 [label=" Yes"]
            q3 -> retain2 [label=" No"]
        }
    """)

    st.markdown(
        """
        **How it works in plain English:**
        1. Imagine asking a series of yes/no questions: "Is the contract month-to-month?"
           → "Is tenure less than 12 months?" → "Are charges above $70?"
           That's a **decision tree** — it keeps splitting until it reaches an answer.
        2. A single tree is easy to overfit (it memorizes the training data too closely).
           So Random Forest builds **200 trees**, each trained on a **random sample**
           of the data and a **random subset** of features.
        3. For a new customer, all 200 trees make their prediction and we take
           the **majority vote**.

        **Why "Random"?** Each tree only sees a random portion of the data and features.
        This diversity prevents the forest from over-relying on any single pattern.

        **Why Label Encoding is fine:** Trees split on thresholds (e.g., "is Contract < 1?").
        They never multiply the encoded number — so the integer values don't introduce false relationships.

        **Strengths:** Handles non-linear patterns, resistant to overfitting, works with Label Encoding.
        **Weaknesses:** Slower than Logistic Regression, less interpretable (200 trees are hard to inspect by hand).
        """
    )

    # ── XGBoost ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. XGBoost (Extreme Gradient Boosting)")
    st.markdown("*The algorithm that wins most Kaggle competitions.*")

    st.graphviz_chart("""
        digraph xgb {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            data [label="Training Data", fillcolor="#dbeafe", color="#3b82f6"]

            t1 [label="Tree 1\\nLearns the main pattern\\n(e.g., contract type)", fillcolor="#d1fae5", color="#10b981"]
            e1 [label="Errors from Tree 1\\n(customers it got wrong)", fillcolor="#fee2e2", color="#ef4444"]
            t2 [label="Tree 2\\nFocuses on Tree 1's mistakes", fillcolor="#d1fae5", color="#10b981"]
            e2 [label="Remaining errors", fillcolor="#fee2e2", color="#ef4444"]
            t3 [label="Tree 3\\nFocuses on remaining mistakes", fillcolor="#d1fae5", color="#10b981"]
            dots [label="... (200 trees total, each fixing\\nthe previous tree's errors)", shape=plaintext, fontname="Helvetica"]
            final [label="Final Prediction\\nSum of all 200 trees\\n(each weighted by learning rate)", fillcolor="#fef3c7", color="#f59e0b"]

            data -> t1 -> e1 -> t2 -> e2 -> t3 -> dots -> final
        }
    """)

    st.markdown(
        """
        **How it works in plain English:**
        1. **Tree 1** tries to predict churn for all customers. It gets some right, some wrong.
        2. **Tree 2** doesn't start from scratch — it specifically focuses on the customers
           that Tree 1 got **wrong**. It tries to correct those mistakes.
        3. **Tree 3** focuses on the remaining mistakes from Tree 1 + Tree 2 combined.
        4. This continues for 200 rounds. Each new tree is a specialist in fixing
           what all previous trees couldn't get right.
        5. The final prediction is the **weighted sum** of all 200 trees.

        **The key difference from Random Forest:**
        - Random Forest: 200 trees trained **independently** (in parallel), then vote.
        - XGBoost: 200 trees trained **sequentially**, each one learning from the previous one's errors.

        **Why "Gradient Boosting"?** "Gradient" refers to the mathematical technique used to
        determine *how* each new tree should focus on errors. It's the same gradient descent
        concept used in deep learning.

        **Strengths:** Usually the most accurate model, handles complex non-linear patterns,
        built-in regularization prevents overfitting.
        **Weaknesses:** Slower to train, harder to interpret, more hyperparameters to tune.
        """
    )

    # ── SHAP ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. SHAP — How We Explain Predictions")
    st.markdown("*Making the black box transparent.*")

    st.graphviz_chart("""
        digraph shap {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            base [label="Base Rate\\n26.5% churn\\n(average customer)", fillcolor="#e0e7ff", color="#6366f1"]
            f1 [label="Contract =\\nMonth-to-month\\n+18%", fillcolor="#fee2e2", color="#ef4444"]
            f2 [label="tenure =\\n2 months\\n+12%", fillcolor="#fee2e2", color="#ef4444"]
            f3 [label="TechSupport =\\nNo\\n+5%", fillcolor="#fee2e2", color="#ef4444"]
            f4 [label="TotalCharges =\\n$150 (low)\\n+3%", fillcolor="#fee2e2", color="#ef4444"]
            f5 [label="Partner = Yes\\n-4%", fillcolor="#d1fae5", color="#10b981"]
            pred [label="Final Prediction\\n60.5% churn", fillcolor="#fef3c7", color="#f59e0b"]

            base -> f1 -> f2 -> f3 -> f4 -> f5 -> pred
        }
    """)

    st.markdown(
        """
        **What SHAP does in plain English:**

        Every prediction starts from the **base rate** — the overall churn rate in the data (~26.5%).
        Then SHAP shows how each feature **pushes** that prediction up or down:

        - Month-to-month contract → pushes probability **up** (toward churn)
        - Low tenure → pushes probability **up** (new customer, high risk)
        - Having a partner → pushes probability **down** (slightly protective)
        - Each feature gets a + or - contribution, and they all add up to the final prediction.

        **Why this matters for business:** If the model predicts an 80% churn probability,
        SHAP tells you *why* — "It's mainly because they're on a month-to-month contract
        and only been with us for 2 months." That's actionable: offer them a yearly contract
        with a discount.

        **The name "SHAP"** stands for SHapley Additive exPlanations, based on Shapley values
        from game theory — a mathematically rigorous way to fairly distribute credit among features.
        """
    )

    # ── Comparison Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Quick Comparison")
    st.markdown(
        """
        | | Logistic Regression | Random Forest | XGBoost |
        |---|---|---|---|
        | **How it learns** | Finds the best weights for a linear equation | Builds many independent trees and averages their votes | Builds trees sequentially, each correcting the last |
        | **Encoding** | One-Hot (needs binary columns) | Label (integers are fine) | Label (integers are fine) |
        | **Speed** | Very fast | Moderate | Slower |
        | **Accuracy** | Good baseline | Very good | Usually best |
        | **Interpretability** | High (weights = feature importance) | Medium (feature importance available) | Medium (needs SHAP for full explanation) |
        | **Best for** | Simple, linear relationships | Non-linear patterns with moderate data | Complex patterns, competitions, production systems |
        """
    )

# ── Tab 0: Predict and Compare ───────────────────────────────────────────────
with tab_predict:
    st.subheader("Predict and Compare — Live Demo")
    st.markdown(
        "Below are **5 real customers** from the test set (data the models have never seen during training). "
        "Click **Run Predictions** to see what each model thinks — then compare against what actually happened."
    )

    raw_df_pred = load_raw_data()

    churned_idx = y_test[y_test == 1].index[:3]
    retained_idx = y_test[y_test == 0].index[:2]
    demo_idx = churned_idx.tolist() + retained_idx.tolist()

    display_columns = [
        "customerID", "gender", "SeniorCitizen", "tenure", "Contract",
        "InternetService", "MonthlyCharges", "TotalCharges",
    ]
    demo_display = raw_df_pred.loc[demo_idx, display_columns].copy()
    demo_display.index = range(1, len(demo_display) + 1)
    demo_display.index.name = "#"

    st.markdown("#### Customer Profiles")
    st.dataframe(demo_display, use_container_width=True)

    if st.button("Run Predictions", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("#### Prediction Results")

        model_names = list(all_models.keys())
        results_rows = []
        for pos, idx in enumerate(demo_idx, 1):
            actual_val = y_test.loc[idx]
            actual_label = "Churned" if actual_val == 1 else "Retained"

            row_result = {
                "#": pos,
                "Customer ID": raw_df_pred.loc[idx, "customerID"],
            }

            all_correct = True
            all_wrong = True
            for model_name in model_names:
                model_obj = all_models[model_name]
                x_data = model_test_data[model_name]
                row_enc = x_data.loc[[idx]]
                pred = model_obj.predict(row_enc)[0]
                proba = model_obj.predict_proba(row_enc)[0][1]
                pred_label = "Churned" if pred == 1 else "Retained"
                correct = pred == actual_val
                if correct:
                    all_wrong = False
                else:
                    all_correct = False
                row_result[f"{model_name}"] = f"{pred_label} ({proba:.0%})"
                row_result[f"{model_name}_correct"] = correct

            row_result["Actual"] = actual_label

            if all_correct:
                row_result["_status"] = "all_correct"
            elif all_wrong:
                row_result["_status"] = "all_wrong"
            else:
                row_result["_status"] = "mixed"

            results_rows.append(row_result)

        results_df = pd.DataFrame(results_rows)

        for _, row in results_df.iterrows():
            status = row["_status"]
            if status == "all_correct":
                icon = "✅"
            elif status == "all_wrong":
                icon = "❌"
            else:
                icon = "⚠️"

            cols = st.columns([0.5, 1.5] + [2] * len(model_names) + [1.2, 0.5])
            cols[0].markdown(f"**{row['#']}**")
            cols[1].markdown(f"`{row['Customer ID']}`")
            for j, mn in enumerate(model_names):
                correct = row[f"{mn}_correct"]
                mark = "✓" if correct else "✗"
                cols[j + 2].markdown(
                    f"{'🟢' if correct else '🔴'} {row[mn]} {mark}"
                )
            cols[len(model_names) + 2].markdown(f"**{row['Actual']}**")
            cols[len(model_names) + 3].markdown(icon)

        st.markdown("---")

        n_total = len(results_rows)
        n_all_correct = sum(1 for r in results_rows if r["_status"] == "all_correct")
        n_all_wrong = sum(1 for r in results_rows if r["_status"] == "all_wrong")
        n_mixed = n_total - n_all_correct - n_all_wrong

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("All Models Correct", f"{n_all_correct} / {n_total}", delta="✅")
        rc2.metric("Mixed Results", f"{n_mixed} / {n_total}", delta="⚠️" if n_mixed > 0 else None)
        rc3.metric("All Models Wrong", f"{n_all_wrong} / {n_total}", delta="❌" if n_all_wrong > 0 else None, delta_color="inverse")

        st.markdown(
            "**Legend:** 🟢 = correct prediction, 🔴 = wrong prediction. "
            "Percentage shown is the model's estimated churn probability."
        )

# ── Tab 1: Model Comparison ──────────────────────────────────────────────────
with tab_compare:
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(3)

    best_model = metrics_df["AUC"].idxmax()
    st.info(f"Best model by AUC: **{best_model}** ({metrics_df.loc[best_model, 'AUC']:.3f})")

    encoding_col = pd.Series({
        "Logistic Regression": "One-Hot",
        "Random Forest": "Label",
        "XGBoost": "Label",
    }, name="Encoding")
    display_metrics = pd.concat([encoding_col, metrics_df], axis=1)

    st.dataframe(
        display_metrics.style.highlight_max(axis=0, subset=metrics_df.columns, color="#c6efce"),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("ROC Curves")
    roc_entries = [(name, all_models[name], model_test_data[name]) for name in all_models]
    st.plotly_chart(plot_roc_curves(roc_entries, y_test), use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(3)
    for i, name in enumerate(all_models):
        with cm_cols[i]:
            y_pred = all_models[name].predict(model_test_data[name])
            st.plotly_chart(
                plot_confusion_matrix(y_test, y_pred, title=name),
                use_container_width=True,
            )

# ── Tab 2: Global SHAP ──────────────────────────────────────────────────────
with tab_shap_global:
    st.subheader("Global Feature Importance — XGBoost")
    st.markdown("SHAP values show how much each feature pushes the prediction toward or away from churn.")

    xgb_model = tree_models["XGBoost"]
    explainer, shap_values = get_shap_explainer(xgb_model, X_train)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax_bar)
    st.pyplot(fig_bar)

    st.markdown("---")
    st.markdown("**Beeswarm Plot** — Each dot is a customer. Color = feature value (red = high, blue = low).")

    fig_bee, ax_bee = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(plt.gcf())

# ── Tab 3: Individual Explanations ───────────────────────────────────────────
with tab_shap_individual:
    st.subheader("Explain a Single Customer's Prediction")

    raw_df = load_raw_data()
    customer_ids = raw_df["customerID"].values
    selected_id = st.selectbox("Select Customer ID", customer_ids[:200])
    idx_in_raw = raw_df[raw_df["customerID"] == selected_id].index[0]

    enc_df, _ = get_encoded_data()

    if idx_in_raw in X_test.index:
        row = X_test.loc[[idx_in_raw]]
        actual = y_test.loc[idx_in_raw]
    elif idx_in_raw in X_train.index:
        row = X_train.loc[[idx_in_raw]]
        actual = y_train.loc[idx_in_raw]
    else:
        row = enc_df.loc[[idx_in_raw], feature_cols]
        actual = enc_df.loc[idx_in_raw, "Churn"]

    proba = xgb_model.predict_proba(row)[0][1]

    c1, c2 = st.columns(2)
    c1.metric("Predicted Churn Probability", f"{proba:.1%}")
    c2.metric("Actual Outcome", "Churned" if actual == 1 else "Retained")

    st.markdown("**Customer Details (raw values):**")
    st.dataframe(raw_df[raw_df["customerID"] == selected_id].T, use_container_width=True)

    st.markdown("---")
    st.markdown("**SHAP Waterfall — Why this prediction?**")
    sv = get_shap_single(explainer, row)
    fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    st.pyplot(plt.gcf())

# ── Tab 4: What-If Predictor ─────────────────────────────────────────────────
with tab_whatif:
    st.subheader("What-If Predictor")
    st.markdown("Adjust customer features and see how churn probability changes in real time.")

    raw_df = load_raw_data()

    wi_col1, wi_col2, wi_col3 = st.columns(3)

    with wi_col1:
        wi_gender = st.selectbox("Gender", ["Female", "Male"], key="wi_gender")
        wi_senior = st.selectbox("Senior Citizen", [0, 1], key="wi_senior")
        wi_partner = st.selectbox("Partner", ["Yes", "No"], key="wi_partner")
        wi_dependents = st.selectbox("Dependents", ["Yes", "No"], key="wi_dep")
        wi_tenure = st.slider("Tenure (months)", 0, 72, 12, key="wi_tenure")

    with wi_col2:
        wi_phone = st.selectbox("Phone Service", ["Yes", "No"], key="wi_phone")
        wi_multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="wi_multi")
        wi_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="wi_inet")
        wi_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="wi_sec")
        wi_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="wi_bak")

    with wi_col3:
        wi_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="wi_prot")
        wi_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="wi_sup")
        wi_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="wi_tv")
        wi_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="wi_mov")
        wi_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="wi_con")

    wi_col4, wi_col5 = st.columns(2)
    with wi_col4:
        wi_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="wi_paper")
        wi_payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="wi_pay",
        )
    with wi_col5:
        wi_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5, key="wi_monthly")
        wi_total = st.slider("Total Charges ($)", 18.0, 9000.0, 1500.0, step=10.0, key="wi_total")

    input_dict = {
        "gender": wi_gender, "Partner": wi_partner, "Dependents": wi_dependents,
        "PhoneService": wi_phone, "MultipleLines": wi_multi, "InternetService": wi_internet,
        "OnlineSecurity": wi_security, "OnlineBackup": wi_backup,
        "DeviceProtection": wi_protection, "TechSupport": wi_support,
        "StreamingTV": wi_tv, "StreamingMovies": wi_movies, "Contract": wi_contract,
        "PaperlessBilling": wi_paperless, "PaymentMethod": wi_payment,
    }
    numeric_dict = {
        "SeniorCitizen": wi_senior, "tenure": wi_tenure,
        "MonthlyCharges": wi_monthly, "TotalCharges": wi_total,
    }

    _, enc_map = get_encoded_data()
    encoded_input = {}
    for col, val in input_dict.items():
        le = enc_map[col]
        if val in le.classes_:
            encoded_input[col] = le.transform([val])[0]
        else:
            encoded_input[col] = 0

    encoded_input.update(numeric_dict)
    input_row = pd.DataFrame([encoded_input])[feature_cols]

    wi_proba = xgb_model.predict_proba(input_row)[0][1]

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Churn Probability", f"{wi_proba:.1%}")
        if wi_proba < 0.3:
            st.success("Low risk — customer likely to stay")
        elif wi_proba < 0.6:
            st.warning("Medium risk — consider retention offer")
        else:
            st.error("High risk — immediate intervention recommended")
    with res_col2:
        st.plotly_chart(plot_gauge(wi_proba), use_container_width=True)

    st.markdown("---")
    st.markdown("**Top Feature Drivers for This Configuration:**")
    sv_wi = get_shap_single(explainer, input_row)
    fig_wi, ax_wi = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(sv_wi[0], max_display=10, show=False)
    st.pyplot(plt.gcf())
