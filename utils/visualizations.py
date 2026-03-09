import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Label"] = counts["Churn"].map({0: "Retained", 1: "Churned"})
    fig = px.pie(
        counts, values="Count", names="Label", hole=0.45,
        color="Label",
        color_discrete_map={"Retained": "#636EFA", "Churned": "#EF553B"},
    )
    fig.update_traces(textinfo="percent+label+value")
    fig.update_layout(title="Customer Churn Distribution", showlegend=False)
    return fig


def plot_feature_histogram(df: pd.DataFrame, feature: str) -> go.Figure:
    temp = df.copy()
    temp["Churn_Label"] = temp["Churn"].map({0: "Retained", 1: "Churned"})
    fig = px.histogram(
        temp, x=feature, color="Churn_Label", barmode="overlay",
        color_discrete_map={"Retained": "#636EFA", "Churned": "#EF553B"},
        opacity=0.7,
    )
    fig.update_layout(title=f"Distribution of {feature} by Churn Status")
    return fig


def plot_categorical_churn_rate(df: pd.DataFrame, feature: str) -> go.Figure:
    grouped = df.groupby(feature)["Churn"].mean().reset_index()
    grouped.columns = [feature, "Churn Rate"]
    grouped["Churn Rate"] = (grouped["Churn Rate"] * 100).round(1)
    fig = px.bar(
        grouped, x=feature, y="Churn Rate",
        text="Churn Rate", color="Churn Rate",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        title=f"Churn Rate by {feature}",
        yaxis_title="Churn Rate (%)",
        coloraxis_showscale=False,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig.update_layout(title="Feature Correlation Heatmap")
    return fig


def plot_roc_curves(model_entries: list, y_test) -> go.Figure:
    """Accept list of (name, model, X_test) tuples so each model can use its own test data."""
    fig = go.Figure()
    for name, model, X_test in model_entries:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        title="ROC Curves — Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Retained", "Churned"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=labels, y=labels, aspect="equal",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        coloraxis_showscale=False,
    )
    return fig


def plot_gauge(probability: float) -> go.Figure:
    color = "#2ecc71" if probability < 0.3 else "#f39c12" if probability < 0.6 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
        },
        title={"text": "Churn Probability"},
    ))
    fig.update_layout(height=300)
    return fig


def plot_segments(X_2d: np.ndarray, cluster_labels: np.ndarray, churn: np.ndarray) -> go.Figure:
    seg_df = pd.DataFrame({
        "UMAP_1": X_2d[:, 0],
        "UMAP_2": X_2d[:, 1],
        "Cluster": cluster_labels.astype(str),
        "Churn": np.where(churn == 1, "Churned", "Retained"),
    })
    fig = px.scatter(
        seg_df, x="UMAP_1", y="UMAP_2",
        color="Cluster", symbol="Churn",
        opacity=0.6,
        symbol_map={"Churned": "x", "Retained": "circle"},
    )
    fig.update_layout(
        title="Customer Segments (UMAP Projection)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
    )
    return fig


def plot_metric_history(history: dict[str, list[float]], batch_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for metric_name, values in history.items():
        fig.add_trace(go.Scatter(
            x=batch_labels[:len(values)], y=values,
            mode="lines+markers", name=metric_name,
        ))
    fig.update_layout(
        title="Model Performance Over Streaming Batches",
        xaxis_title="Batch",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
    )
    return fig
