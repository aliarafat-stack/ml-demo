import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

from utils.data_loader import get_train_test
from utils.visualizations import plot_metric_history, plot_confusion_matrix

st.set_page_config(page_title="Live Updates", page_icon="⚡", layout="wide")
st.title("Live Model Updates — Incremental Learning")
st.markdown("---")

tab_explain, tab_demo = st.tabs(["How It Works", "Live Demo"])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: How It Works — Full explanation with diagrams
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("The Problem: Models Go Stale")
    st.markdown(
        """
        In the real world, a model trained today will slowly become **less accurate** over time.
        Customer behavior changes, new products launch, the economy shifts.
        This is called **model drift** — the patterns the model learned no longer match reality.
        """
    )

    st.graphviz_chart("""
        digraph drift {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            jan [label="Jan 2024\\nModel trained\\nAccuracy: 82%", fillcolor="#d1fae5", color="#10b981"]
            jun [label="Jun 2024\\nSame model\\nAccuracy: 76%", fillcolor="#fef3c7", color="#f59e0b"]
            dec [label="Dec 2024\\nSame model\\nAccuracy: 68%", fillcolor="#fee2e2", color="#ef4444"]
            bad [label="Model is now\\nmaking costly\\nwrong predictions", fillcolor="#fee2e2", color="#ef4444", shape=ellipse]

            jan -> jun [label="  drift"]
            jun -> dec [label="  more drift"]
            dec -> bad
        }
    """)

    st.markdown("---")
    st.subheader("Solution 1: Full Retraining (Expensive)")

    st.graphviz_chart("""
        digraph retrain {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            all [label="ALL Data\\n(old + new)\\n→ millions of rows", fillcolor="#dbeafe", color="#3b82f6"]
            train [label="Retrain from scratch\\n→ hours/days of compute\\n→ model downtime", fillcolor="#fee2e2", color="#ef4444"]
            deploy [label="Deploy\\nnew model", fillcolor="#d1fae5", color="#10b981"]

            all -> train -> deploy
        }
    """)

    st.markdown(
        """
        The brute-force approach: collect all data (historical + new), retrain the model completely.

        **Problems:**
        - **Slow** — retraining on millions of rows can take hours
        - **Expensive** — requires significant compute resources
        - **Downtime** — the model is unavailable during retraining
        - **Wasteful** — most of the data hasn't changed, only new records were added
        """
    )

    st.markdown("---")
    st.subheader("Solution 2: Incremental Learning with partial_fit (Smart)")

    st.graphviz_chart("""
        digraph partial {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            existing [label="Existing Model\\n(already trained)", fillcolor="#d1fae5", color="#10b981"]
            batch [label="New Batch\\n(50–500 rows)\\njust arrived", fillcolor="#dbeafe", color="#3b82f6"]
            update [label="partial_fit()\\nUpdate weights\\n→ seconds", fillcolor="#fef3c7", color="#f59e0b"]
            better [label="Updated Model\\nNow knows the\\nnew patterns too", fillcolor="#d1fae5", color="#10b981"]

            existing -> update
            batch -> update
            update -> better
        }
    """)

    st.markdown(
        """
        Instead of starting over, we **update** the existing model with only the new data.
        The model adjusts its weights slightly to account for the new patterns — without
        forgetting what it already learned.

        **Benefits:**
        - **Fast** — updating takes seconds instead of hours
        - **No downtime** — the model stays live while updating
        - **Memory efficient** — we only need the new data in memory, not the full history
        - **Continuous** — can update after every batch, every hour, or every day
        """
    )

    st.markdown("---")
    st.subheader("How SGDClassifier + partial_fit Works")

    st.markdown(
        """
        **SGD** stands for **Stochastic Gradient Descent**. Here's the intuition:
        """
    )

    st.graphviz_chart("""
        digraph sgd {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            start [label="Model has current weights\\n(its 'knowledge')", fillcolor="#e0e7ff", color="#6366f1"]
            predict [label="Predict churn for\\na batch of customers", fillcolor="#dbeafe", color="#3b82f6"]
            error [label="Calculate error\\n(how wrong was it?)", fillcolor="#fee2e2", color="#ef4444"]
            gradient [label="Calculate gradient\\n(which direction to\\nadjust weights)", fillcolor="#fef3c7", color="#f59e0b"]
            update [label="Nudge weights in\\nthe right direction\\n(small step)", fillcolor="#d1fae5", color="#10b981"]

            start -> predict -> error -> gradient -> update
            update -> start [style=dashed, label="  next batch"]
        }
    """)

    st.markdown(
        """
        **Step by step:**
        1. The model has **weights** — one number per feature that represents "how much does
           this feature contribute to churn?"
        2. A new batch of customers arrives. The model predicts churn for each one.
        3. We check: how wrong was it? The **loss function** (log loss) measures the gap
           between predicted probabilities and actual outcomes.
        4. The **gradient** tells us the direction and magnitude to adjust each weight to
           reduce the error. Think of it like standing on a hill and figuring out which
           direction is downhill.
        5. We take a small step in that direction (the **learning rate** controls the step size).
        6. Repeat for the next batch.

        **Why "Stochastic"?** Instead of computing the gradient on *all* data (expensive),
        SGD computes it on a small batch (stochastic = random sample). This is noisy but
        much faster, and the noise actually helps avoid getting stuck in local minima.
        """
    )

    st.markdown("---")
    st.subheader("Production Architecture — How This Works in the Real World")

    st.graphviz_chart("""
        digraph production {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            subgraph cluster_data {
                label="Data Sources"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"

                web [label="Website\\nActivity", fillcolor="#dbeafe", color="#3b82f6"]
                crm [label="CRM\\nSystem", fillcolor="#dbeafe", color="#3b82f6"]
                billing [label="Billing\\nRecords", fillcolor="#dbeafe", color="#3b82f6"]
            }

            queue [label="Message Queue\\n(Kafka / RabbitMQ)\\nBuffers incoming events", fillcolor="#e0e7ff", color="#6366f1"]

            subgraph cluster_pipeline {
                label="ML Pipeline"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"

                preproc [label="Preprocessing\\nService\\n(encode, scale)", fillcolor="#fce7f3", color="#ec4899"]
                model [label="Live Model\\n(SGDClassifier)\\npartial_fit()", fillcolor="#d1fae5", color="#10b981"]
                monitor [label="Performance\\nMonitor\\n(track accuracy)", fillcolor="#fef3c7", color="#f59e0b"]
            }

            alert [label="Drift Alert\\n(retrain if accuracy\\ndrops below threshold)", fillcolor="#fee2e2", color="#ef4444"]
            api [label="Prediction API\\n/predict → churn score\\nserves real-time requests", fillcolor="#d1fae5", color="#10b981"]

            subgraph cluster_action {
                label="Business Actions"
                style=dashed
                color="#94a3b8"
                fontname="Helvetica"

                dashboard [label="Risk\\nDashboard", fillcolor="#fef3c7", color="#f59e0b"]
                retention [label="Automated\\nRetention Offers", fillcolor="#fef3c7", color="#f59e0b"]
                crm_out [label="CRM\\nAlerts", fillcolor="#fef3c7", color="#f59e0b"]
            }

            web -> queue
            crm -> queue
            billing -> queue
            queue -> preproc -> model
            model -> monitor
            monitor -> alert [style=dashed]
            model -> api
            api -> dashboard
            api -> retention
            api -> crm_out
        }
    """)

    st.markdown(
        """
        **How it fits together in production:**

        | Component | Role | Technology Examples |
        |---|---|---|
        | **Data Sources** | Customer interactions generate events (page visits, support calls, payments) | Databases, APIs, event logs |
        | **Message Queue** | Buffers and orders incoming data events | Apache Kafka, RabbitMQ, AWS SQS |
        | **Preprocessing** | Encodes, scales, and validates new data before it reaches the model | Python microservice, Spark Streaming |
        | **Live Model** | Runs `partial_fit()` on new batches, updates weights in-place | scikit-learn SGDClassifier, River library |
        | **Monitor** | Tracks accuracy, precision, recall over time — detects drift | MLflow, Prometheus + Grafana |
        | **Prediction API** | Serves real-time churn predictions for individual customers | FastAPI, Flask, AWS SageMaker |
        | **Business Actions** | Turns predictions into interventions — offers, alerts, dashboard flags | CRM integration, email triggers |

        **Key insight:** The model is never "done" — it continuously learns from new data while
        simultaneously serving predictions. This is the modern ML deployment pattern.
        """
    )

    st.markdown("---")
    st.subheader("partial_fit vs Full Retraining — When to Use Which")
    st.markdown(
        """
        | Scenario | Best Approach | Why |
        |---|---|---|
        | New data arrives every few hours | **partial_fit** | Fast updates, no downtime |
        | Massive distribution shift (e.g., pandemic) | **Full retrain** | Old patterns are fundamentally different now |
        | Model accuracy drops gradually | **partial_fit** with monitoring | Catches slow drift |
        | Model accuracy drops suddenly | **Full retrain** + investigation | Something fundamental changed |
        | Adding new features to the model | **Full retrain** | Model architecture changed |
        | Same features, new data | **partial_fit** | Exactly what it's designed for |

        In practice, many companies use a **hybrid approach**: `partial_fit` for daily updates,
        scheduled full retraining weekly or monthly as a safety net.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: Live Demo — Streaming Simulation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("Streaming Simulation")
    st.markdown(
        """
        **What happens when you press Start:**
        1. The model is **initially trained** on 60% of the training data
        2. The remaining 40% is split into **mini-batches** that arrive one at a time
        3. After each batch, the model **updates its weights** via `partial_fit()` and we re-evaluate

        Watch the metrics chart update in real-time as the model learns from the stream.
        """
    )

    st.markdown("---")

    X_train_full, X_test, y_train_full, y_test, _, feature_cols = get_train_test()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    n_initial = int(len(X_train_scaled) * 0.6)
    X_initial = X_train_scaled[:n_initial]
    y_initial = y_train_full.values[:n_initial]
    X_stream = X_train_scaled[n_initial:]
    y_stream = y_train_full.values[n_initial:]

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        n_batches = st.slider("Number of streaming batches", 5, 30, 15)
    with col_config2:
        delay = st.slider("Delay between batches (seconds)", 0.1, 2.0, 0.4, step=0.1)

    batch_size = len(X_stream) // n_batches

    st.markdown("---")

    if st.button("Start Stream", type="primary", use_container_width=True):
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        model.partial_fit(X_initial, y_initial, classes=np.array([0, 1]))

        y_pred_init = model.predict(X_test_scaled)
        init_acc = accuracy_score(y_test, y_pred_init)
        init_f1 = f1_score(y_test, y_pred_init, zero_division=0)

        st.info(
            f"Initial model trained on {n_initial:,} samples — "
            f"Accuracy: {init_acc:.3f}, F1: {init_f1:.3f}"
        )

        y_pred_before = y_pred_init.copy()

        history = {"Accuracy": [init_acc], "F1 Score": [init_f1]}
        batch_labels = ["Initial"]

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_container = st.empty()
        metrics_row = st.empty()

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size if i < n_batches - 1 else len(X_stream)
            X_batch = X_stream[start:end]
            y_batch = y_stream[start:end]

            model.partial_fit(X_batch, y_batch)

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            batch_label = f"Batch {i + 1}"
            batch_labels.append(batch_label)
            history["Accuracy"].append(acc)
            history["F1 Score"].append(f1)

            progress_bar.progress((i + 1) / n_batches)
            status_text.markdown(
                f"**{batch_label}:** Ingested {len(X_batch)} new customers "
                f"(total streamed: {end:,} / {len(X_stream):,})"
            )

            with metrics_row.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{acc:.3f}", f"{acc - history['Accuracy'][-2]:+.3f}")
                m2.metric("F1 Score", f"{f1:.3f}", f"{f1 - history['F1 Score'][-2]:+.3f}")
                m3.metric("Samples Processed", f"{n_initial + end:,}")

            with chart_container.container():
                st.plotly_chart(
                    plot_metric_history(history, batch_labels),
                    use_container_width=True,
                    key=f"metric_chart_{i}",
                )

            time.sleep(delay)

        progress_bar.progress(1.0)
        status_text.success(
            f"Streaming complete! Processed {len(X_stream):,} additional customers in {n_batches} batches."
        )

        st.markdown("---")
        st.subheader("Before vs After — Confusion Matrix")
        cm_after = confusion_matrix(y_test, model.predict(X_test_scaled))

        bef, aft = st.columns(2)
        with bef:
            st.plotly_chart(
                plot_confusion_matrix(y_test, y_pred_before, title="After Initial Training"),
                use_container_width=True,
            )
            st.caption(f"Accuracy: {init_acc:.3f}")
        with aft:
            st.plotly_chart(
                plot_confusion_matrix(y_test, model.predict(X_test_scaled), title="After All Batches"),
                use_container_width=True,
            )
            final_acc = history["Accuracy"][-1]
            delta = final_acc - init_acc
            st.caption(f"Accuracy: {final_acc:.3f} ({delta:+.3f})")

        st.markdown("---")
        st.subheader("Summary")
        improvement_acc = history["Accuracy"][-1] - history["Accuracy"][0]
        improvement_f1 = history["F1 Score"][-1] - history["F1 Score"][0]
        st.markdown(
            f"""
            | Metric | Initial | Final | Change |
            |---|---|---|---|
            | Accuracy | {history['Accuracy'][0]:.3f} | {history['Accuracy'][-1]:.3f} | {improvement_acc:+.3f} |
            | F1 Score | {history['F1 Score'][0]:.3f} | {history['F1 Score'][-1]:.3f} | {improvement_f1:+.3f} |
            """
        )
    else:
        st.markdown("Press **Start Stream** to begin the incremental learning simulation.")
