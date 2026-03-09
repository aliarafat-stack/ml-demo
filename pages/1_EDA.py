import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

from utils.data_loader import load_raw_data, CATEGORICAL_COLS, NUMERIC_COLS
from utils.visualizations import (
    plot_churn_distribution,
    plot_feature_histogram,
    plot_categorical_churn_rate,
    plot_correlation_heatmap,
    plot_segments,
)

st.set_page_config(page_title="EDA — Churn Demo", page_icon="🔍", layout="wide")
st.title("Exploratory Data Analysis")
st.markdown(
    "EDA is the first step in any data science project. Before building models, we need to "
    "**look at the data**, understand its shape, spot patterns, and form hypotheses about "
    "what drives customer churn."
)

with st.expander("Why do EDA? What's the goal here?", expanded=False):
    st.graphviz_chart("""
        digraph eda_flow {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            raw   [label="Raw Data\\n7,043 rows × 21 cols", fillcolor="#dbeafe", color="#3b82f6"]
            ask   [label="Ask Questions\\nWho churns? Why?", fillcolor="#e0e7ff", color="#6366f1"]
            viz   [label="Visualize\\nCharts & Patterns", fillcolor="#fce7f3", color="#ec4899"]
            hypo  [label="Form Hypotheses\\nContract type matters!", fillcolor="#d1fae5", color="#10b981"]
            model [label="Guide Modeling\\nPick features & algos", fillcolor="#fef3c7", color="#f59e0b"]

            raw -> ask -> viz -> hypo -> model
        }
    """)
    st.markdown(
        """
        **Imagine you're a detective.** You've been handed a case file (the dataset) and
        told "some customers are leaving — figure out why." You wouldn't jump straight to
        conclusions. You'd:

        1. **Look at what you have** — How many customers? What info do we know about them?
        2. **Spot the obvious** — What percentage are leaving? Is the problem big or small?
        3. **Look for clues** — Do short-tenure customers leave more? Do expensive plans cause churn?
        4. **Form theories** — "Month-to-month customers with fiber optic and electronic check payments are the highest risk"
        5. **Use those theories to build your case** — These hypotheses guide which features matter most for our model

        **Without EDA, we'd be building models blindly.** We might include useless features
        (like gender, which has no effect on churn) or miss critical ones (like contract type,
        which is the #1 predictor).

        Each chart below answers a specific question about the data. Read the explanations
        below each chart — they translate the visual into a business insight.
        """
    )

st.markdown("---")

df = load_raw_data()

# ── Dataset Preview ──────────────────────────────────────────────────────────
st.subheader("Dataset Preview")
st.markdown(
    "Each row is one customer. Each column is a piece of information about them — "
    "who they are, what services they use, how much they pay, and whether they left (churned)."
)
col_filter, col_churn = st.columns([3, 1])
with col_churn:
    churn_filter = st.selectbox("Filter by Churn", ["All", "Churned", "Retained"])
filtered = df.copy()
if churn_filter == "Churned":
    filtered = filtered[filtered["Churn"] == 1]
elif churn_filter == "Retained":
    filtered = filtered[filtered["Churn"] == 0]
st.dataframe(filtered.head(200), use_container_width=True, height=300)
st.caption(f"Showing {min(200, len(filtered)):,} of {len(filtered):,} rows")

st.markdown("---")

# ── Churn Distribution ───────────────────────────────────────────────────────
st.subheader("Churn Distribution")
st.markdown(
    "**What this chart shows:** A donut chart splitting all customers into two groups — "
    "those who left (churned) and those who stayed (retained)."
)
st.plotly_chart(plot_churn_distribution(df), use_container_width=True)

churned_count = int(df["Churn"].sum())
retained_count = int(len(df) - churned_count)
churn_pct = df["Churn"].mean() * 100

st.info(
    f"**Key takeaway:** About **1 in 4 customers left** — that's a **{churn_pct:.1f}% churn rate**. "
    f"Out of {len(df):,} total customers, we lost **{churned_count:,}** and retained **{retained_count:,}**. "
    f"This is a significant problem: if each churned customer was paying ~$65/month, "
    f"that's roughly **${churned_count * 65 * 12:,.0f} in lost annual revenue**."
)

st.markdown("---")

# ── Numeric Feature Distributions ────────────────────────────────────────────
st.subheader("Numeric Feature Distributions")
st.markdown(
    "**What this chart shows:** A histogram — it counts how many customers fall into each "
    "range of values for a numeric feature. The blue bars are retained customers, the red "
    "bars are churned customers. Where you see more red, churn is concentrated."
)

num_feature = st.selectbox(
    "Select a numeric feature",
    ["tenure", "MonthlyCharges", "TotalCharges"],
)
st.plotly_chart(plot_feature_histogram(df, num_feature), use_container_width=True)

histogram_explanations = {
    "tenure": (
        "**Tenure** = how many months the customer has been with us (0 = just signed up, 72 = 6 years).\n\n"
        "**What we see:** The red bars (churned customers) pile up heavily on the **left side** (0–10 months). "
        "This means **new customers are the most likely to leave**. Once a customer survives past "
        "roughly 15–20 months, the red bars thin out dramatically — they tend to stick around.\n\n"
        "**Business insight:** The first year is the danger zone. Retention efforts (welcome calls, "
        "onboarding support, loyalty discounts) should focus on customers in their first 12 months."
    ),
    "MonthlyCharges": (
        "**Monthly Charges** = what the customer pays each month ($18 to $118).\n\n"
        "**What we see:** Churned customers (red) are concentrated in the **$70–$110 range** — "
        "the high end. Customers paying ~$20/month rarely churn. This suggests that "
        "**customers on expensive plans leave more often**, possibly because they feel they're "
        "not getting enough value for the price.\n\n"
        "**Business insight:** High-paying customers need extra attention — premium support, "
        "exclusive features, or periodic check-ins to ensure they feel the price is justified."
    ),
    "TotalCharges": (
        "**Total Charges** = the cumulative amount the customer has paid over their entire tenure.\n\n"
        "**What we see:** Churned customers cluster at the **low end** ($0–$500). This makes sense — "
        "they left early, so they haven't accumulated much total spend. Long-term retained customers "
        "spread across a wide range up to $8,000+.\n\n"
        "**Business insight:** This confirms the tenure pattern — customers who churn do so quickly, "
        "before they've invested much financially. The more a customer has paid cumulatively, "
        "the more \"sunk cost\" keeps them, making them less likely to leave."
    ),
}
st.markdown(histogram_explanations[num_feature])

st.markdown("---")

# ── Categorical Churn Rates ──────────────────────────────────────────────────
st.subheader("Churn Rate by Category")
st.markdown(
    "**What this chart shows:** For each value of a categorical feature, the bar height shows "
    "what **percentage** of customers in that group churned. Taller bars = higher churn risk. "
    "The color scale goes from green (low churn) to red (high churn)."
)

cat_feature = st.selectbox(
    "Select a categorical feature",
    ["Contract", "InternetService", "PaymentMethod", "TechSupport",
     "OnlineSecurity", "gender", "Partner", "Dependents"],
)
st.plotly_chart(plot_categorical_churn_rate(df, cat_feature), use_container_width=True)

categorical_explanations = {
    "Contract": (
        "**This is the single strongest predictor of churn in our data.**\n\n"
        "- **Month-to-month** customers churn at ~42% — they have no commitment and can leave any time.\n"
        "- **One-year** contract customers churn at ~11% — some commitment helps.\n"
        "- **Two-year** contract customers churn at only ~3% — long-term lock-in virtually eliminates churn.\n\n"
        "**Business insight:** Incentivizing customers to move from month-to-month to annual contracts "
        "(e.g., a 10% discount for committing to a year) could dramatically reduce churn."
    ),
    "InternetService": (
        "- **Fiber optic** customers churn at ~42%, despite being on the premium service.\n"
        "- **DSL** customers churn at ~19%.\n"
        "- Customers with **no internet** churn at only ~7%.\n\n"
        "**Why?** Fiber optic is expensive and may attract price-sensitive customers who switch "
        "to competitors. They also have higher expectations for speed and reliability.\n\n"
        "**Business insight:** Fiber optic customers need proactive quality-of-service monitoring "
        "and competitive pricing reviews."
    ),
    "PaymentMethod": (
        "- **Electronic check** users churn at ~45% — the highest by far.\n"
        "- All other methods (mailed check, bank transfer, credit card) churn at ~15–18%.\n\n"
        "**Why?** Electronic check requires active effort each month (no auto-deduction), "
        "so these customers are less \"locked in\" and may forget or choose not to pay.\n\n"
        "**Business insight:** Encourage electronic check users to switch to auto-pay "
        "(bank transfer or credit card) with a small discount."
    ),
    "TechSupport": (
        "- Customers **without** tech support churn at ~42%.\n"
        "- Customers **with** tech support churn at only ~15%.\n\n"
        "**Why?** When something breaks and there's no support, frustration drives customers away. "
        "Having tech support means problems get resolved, increasing satisfaction.\n\n"
        "**Business insight:** Bundling free or discounted tech support with plans could be "
        "a cost-effective retention tool."
    ),
    "OnlineSecurity": (
        "- Customers **without** online security churn at ~42%.\n"
        "- Customers **with** online security churn at only ~15%.\n\n"
        "**Similar to TechSupport** — add-on services increase the customer's investment "
        "in the platform and provide more value, making them less likely to leave.\n\n"
        "**Business insight:** The more services a customer uses, the \"stickier\" they become. "
        "Consider bundling security with base plans."
    ),
    "gender": (
        "- Male and Female customers churn at **nearly identical rates** (~26–27%).\n\n"
        "**Gender is not a useful predictor of churn** in this dataset. "
        "The model will learn to mostly ignore it.\n\n"
        "**Business insight:** Retention strategies don't need to be gender-specific."
    ),
    "Partner": (
        "- Customers **without** a partner churn at ~33%.\n"
        "- Customers **with** a partner churn at ~20%.\n\n"
        "**Why?** Customers with partners may have shared accounts or plans, "
        "making switching more inconvenient.\n\n"
        "**Business insight:** Family or couples plans could improve retention for single customers."
    ),
    "Dependents": (
        "- Customers **without** dependents churn at ~31%.\n"
        "- Customers **with** dependents churn at ~15%.\n\n"
        "**Why?** Families with children using the service (streaming, internet) are less "
        "likely to disrupt everyone by switching providers.\n\n"
        "**Business insight:** Family bundles and kid-friendly features can boost retention."
    ),
}
st.markdown(categorical_explanations.get(cat_feature, ""))

st.markdown("---")

# ── Correlation Heatmap ──────────────────────────────────────────────────────
st.subheader("Correlation Heatmap")
st.markdown(
    "**What this chart shows:** Correlation measures how two features move together, "
    "on a scale from **-1** (perfect opposite) to **+1** (perfect match). "
    "A value near **0** means no relationship.\n\n"
    "Think of it this way: if you know one value, how well can you guess the other?"
)

corr_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
st.plotly_chart(plot_correlation_heatmap(df, corr_cols), use_container_width=True)

corr_matrix = df[corr_cols].corr()
tenure_total = corr_matrix.loc["tenure", "TotalCharges"]
monthly_churn = corr_matrix.loc["MonthlyCharges", "Churn"]
tenure_churn = corr_matrix.loc["tenure", "Churn"]

st.markdown(
    f"""
    **Key relationships in our data:**

    | Feature Pair | Correlation | What it means |
    |---|---|---|
    | tenure vs TotalCharges | **{tenure_total:.2f}** (strong positive) | The longer a customer stays, the more they pay in total — obvious but confirms data quality |
    | MonthlyCharges vs Churn | **{monthly_churn:.2f}** (slight positive) | Higher monthly bills are weakly associated with higher churn — expensive plans drive some customers away |
    | tenure vs Churn | **{tenure_churn:.2f}** (moderate negative) | Longer-tenured customers are less likely to churn — loyalty grows over time |

    **Reading the colors:** Deep red = strong positive correlation. Deep blue = strong negative. White/light = little or no relationship.
    """
)

st.markdown("---")

# ── Customer Segmentation (UMAP + KMeans) ────────────────────────────────────
st.subheader("Customer Segments (KMeans Clustering)")
st.markdown(
    "**What this chart shows:** We used **KMeans** to automatically group "
    "customers into clusters based on their tenure, monthly charges, total charges, and senior "
    "citizen status. Then we projected the groups onto a 2D scatter plot using **UMAP** "
    "so we can visualize them.\n\n"
    "- Each **dot** is one customer.\n"
    "- **Colors** represent different clusters (groups of similar customers).\n"
    "- **Circles** (○) = customers who stayed. **X marks** (✕) = customers who churned.\n\n"
    "Look for clusters where X marks are concentrated — those are the high-risk customer segments."
)

with st.expander("How does KMeans work? (step-by-step with diagram)", expanded=False):
    st.graphviz_chart("""
        digraph kmeans {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            input [label="Input: 7,043 customers\\neach with 4 features\\n(tenure, monthly, total, senior)", fillcolor="#dbeafe", color="#3b82f6"]
            scale [label="Step 0: Scale features\\nso all are on equal footing\\n(StandardScaler)", fillcolor="#e0e7ff", color="#6366f1"]
            init [label="Step 1: Pick k random\\npoints as initial\\ncluster centers (centroids)", fillcolor="#fce7f3", color="#ec4899"]
            assign [label="Step 2: ASSIGN\\nEach customer goes to the\\nnearest centroid\\n(Euclidean distance)", fillcolor="#d1fae5", color="#10b981"]
            update [label="Step 3: UPDATE\\nMove each centroid to the\\naverage of its assigned\\ncustomers", fillcolor="#fef3c7", color="#f59e0b"]
            check [label="Step 4: Check\\nDid centroids move?", fillcolor="#fee2e2", color="#ef4444"]
            done [label="Done!\\nEach customer has\\na cluster label", fillcolor="#d1fae5", color="#10b981"]

            input -> scale -> init -> assign -> update -> check
            check -> assign [label="Yes → repeat", style=dashed]
            check -> done [label="No → converged"]
        }
    """)

    st.markdown(
        """
        **The algorithm in plain English:**

        1. **Scale the features** — Tenure ranges 0–72, TotalCharges ranges 0–8,685. Without scaling,
           TotalCharges would dominate the distance calculation just because its numbers are bigger.
           StandardScaler puts all features on the same scale.

        2. **Pick starting points** — Randomly place k "centroids" (cluster centers) in the data space.
           Each centroid is a point with 4 coordinates (one per feature).

        3. **Assign customers** — For each of the 7,043 customers, calculate the distance to every centroid.
           Assign the customer to the **nearest** one. Distance is measured using **Euclidean distance**:
        """
    )
    st.latex(r"d = \sqrt{(tenure_1 - tenure_2)^2 + (monthly_1 - monthly_2)^2 + (total_1 - total_2)^2 + (senior_1 - senior_2)^2}")
    st.markdown(
        """
        4. **Update centroids** — Each centroid moves to the **average position** of all customers
           assigned to it. If Cluster 0 has 2,000 customers with average tenure=40, monthly=$45,
           total=$3,200, senior=0.1, the centroid moves to (40, 45, 3200, 0.1).

        5. **Repeat** steps 3–4 until the centroids stop moving (convergence). Typically takes 5–20 iterations.

        **What determines a cluster?** Customers end up in the same cluster because they are
        **similar across all 4 features simultaneously**. A cluster isn't just "high tenure"
        or "low charges" — it's a combination: "long-tenure, medium-charge, non-senior customers."
        """
    )

with st.expander("What are UMAP Dimension 1 and Dimension 2?", expanded=False):
    st.markdown(
        """
        Each customer has **4 features** — but our screen is only 2D.
        **UMAP** squashes 4 dimensions down to 2 while keeping similar customers close together.

        **Dimension 1** and **Dimension 2** are artificial axes — not any single real feature.
        The axis numbers have no meaningful unit. What matters is **relative position**:
        dots close together = similar customers, dots far apart = different customers.

        **Analogy:** UMAP is like taking a photo of 4D data from the best angle so that
        clusters are visible on a flat image.
        """
    )


@st.cache_data
def compute_segments(data: pd.DataFrame, n_clusters: int = 4):
    features = data[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(scaled)
    return embedding, labels, kmeans, scaler


n_clusters = st.slider("Number of clusters", 2, 8, 4)
embedding, cluster_labels, kmeans_model, seg_scaler = compute_segments(df, n_clusters)
st.plotly_chart(
    plot_segments(embedding, cluster_labels, df["Churn"].values),
    use_container_width=True,
)

# ── Cluster interpretation ───────────────────────────────────────────────────
st.markdown("### What Each Cluster Represents")
st.markdown(
    "Each cluster is a group of customers who are **similar across all 4 features**. "
    "Let's look at the average profile of each cluster to understand who they are."
)

cluster_stats = []
cluster_profiles = []
for c in range(n_clusters):
    mask = cluster_labels == c
    cluster_df = df[mask]
    avg_tenure = cluster_df["tenure"].mean()
    avg_monthly = cluster_df["MonthlyCharges"].mean()
    avg_total = cluster_df["TotalCharges"].mean()
    churn_rate = cluster_df["Churn"].mean()
    senior_pct = cluster_df["SeniorCitizen"].mean()

    if avg_tenure < 20 and churn_rate > 0.3:
        label = "New & High-Risk"
    elif avg_tenure > 40 and churn_rate < 0.15:
        label = "Loyal Veterans"
    elif avg_monthly > 70 and churn_rate > 0.25:
        label = "Premium At-Risk"
    elif avg_monthly < 40:
        label = "Budget Customers"
    elif senior_pct > 0.3:
        label = "Senior Segment"
    else:
        label = "Mid-Tier"

    cluster_stats.append({
        "Cluster": f"{c} — {label}",
        "Size": f"{int(mask.sum()):,}",
        "Churn Rate": f"{churn_rate:.1%}",
        "Avg Tenure": f"{avg_tenure:.0f} months",
        "Avg Monthly": f"${avg_monthly:.0f}",
        "Avg Total": f"${avg_total:,.0f}",
        "Senior %": f"{senior_pct:.0%}",
    })
    cluster_profiles.append({
        "cluster": c, "label": label, "churn_rate": churn_rate,
        "tenure": avg_tenure, "monthly": avg_monthly, "size": int(mask.sum()),
    })

st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True, hide_index=True)

highest_churn = max(cluster_profiles, key=lambda x: x["churn_rate"])
lowest_churn = min(cluster_profiles, key=lambda x: x["churn_rate"])

st.markdown(
    f"""
    **Interpreting the clusters:**

    - **Highest churn cluster:** Cluster {highest_churn['cluster']} ({highest_churn['label']}) at
      **{highest_churn['churn_rate']:.0%}** churn — these {highest_churn['size']:,} customers have
      avg tenure of {highest_churn['tenure']:.0f} months and pay ${highest_churn['monthly']:.0f}/month.
      **These are the customers we should target first with retention offers.**

    - **Lowest churn cluster:** Cluster {lowest_churn['cluster']} ({lowest_churn['label']}) at
      **{lowest_churn['churn_rate']:.0%}** churn — these {lowest_churn['size']:,} customers have
      avg tenure of {lowest_churn['tenure']:.0f} months and pay ${lowest_churn['monthly']:.0f}/month.
      **These are our most loyal customers — study what keeps them happy.**
    """
)

with st.expander("How do clusters help us understand the data?", expanded=False):
    st.markdown(
        """
        **Without clustering:** We know the overall churn rate is ~26.5%. But that's an average —
        it hides the fact that some customer groups churn at 40%+ while others churn at <10%.

        **With clustering:** We can see that the problem isn't uniform. Churn is concentrated
        in specific segments. This unlocks targeted strategies:

        | Cluster Type | Churn | Strategy |
        |---|---|---|
        | New, high-charge customers | Very high | Onboarding program, welcome discount, proactive support |
        | Long-tenure, moderate charge | Very low | Loyalty rewards, referral program |
        | Premium plan, short tenure | High | Dedicated account manager, premium support |
        | Budget customers | Low-moderate | Upsell opportunities, bundle discounts |

        **The business value:** Instead of one generic retention campaign for all 7,043 customers
        (expensive, low hit rate), we can run targeted campaigns for specific clusters
        (cheaper, higher conversion rate).
        """
    )

# ── Centroid visualization ───────────────────────────────────────────────────
st.markdown("### Cluster Centroids — The \"Average Customer\" in Each Group")
st.markdown(
    "The centroid is the center point of each cluster — think of it as the "
    "\"typical customer\" in that group. Here's what each centroid looks like in the original feature space:"
)

centroid_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
centroids_original = seg_scaler.inverse_transform(kmeans_model.cluster_centers_)
centroid_df = pd.DataFrame(centroids_original, columns=centroid_features)
centroid_df.insert(0, "Cluster", [f"Cluster {i}" for i in range(n_clusters)])
centroid_df["tenure"] = centroid_df["tenure"].round(1)
centroid_df["MonthlyCharges"] = centroid_df["MonthlyCharges"].round(2)
centroid_df["TotalCharges"] = centroid_df["TotalCharges"].round(0)
centroid_df["SeniorCitizen"] = centroid_df["SeniorCitizen"].round(2)
st.dataframe(centroid_df, use_container_width=True, hide_index=True)

import plotly.graph_objects as go_cent
fig_radar = go_cent.Figure()
categories = centroid_features + [centroid_features[0]]
for i in range(n_clusters):
    vals = kmeans_model.cluster_centers_[i].tolist()
    vals.append(vals[0])
    fig_radar.add_trace(go_cent.Scatterpolar(
        r=vals, theta=categories, fill="toself", name=f"Cluster {i}",
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Centroid Profiles (scaled features — 0 = average, positive = above average)",
    height=400,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown(
    "**Reading the radar chart:** Each axis is one feature (scaled). "
    "If a cluster extends far on \"tenure\" and far on \"TotalCharges\" but stays close to center "
    "on \"MonthlyCharges\", those are long-tenure customers with moderate monthly bills but high "
    "cumulative spend."
)
