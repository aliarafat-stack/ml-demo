import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.rec_data_loader import load_raw_transactions, load_clean_transactions, build_interaction_matrix

st.set_page_config(page_title="Recommendation EDA", page_icon="🛒", layout="wide")
st.title("Product Recommendation — Exploratory Data Analysis")

with st.expander("Why do EDA for a recommendation system?", expanded=False):
    st.graphviz_chart("""
        digraph eda_rec {
            rankdir=LR
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            txn [label="Transaction Data\\n541K purchases", fillcolor="#dbeafe", color="#3b82f6"]
            who [label="Who buys what?\\nPurchase patterns", fillcolor="#e0e7ff", color="#6366f1"]
            sparse [label="How sparse is it?\\nCan we recommend?", fillcolor="#fce7f3", color="#ec4899"]
            pop [label="Popular vs Niche\\nProduct distribution", fillcolor="#d1fae5", color="#10b981"]
            ready [label="Ready for\\nMatrix Factorization", fillcolor="#fef3c7", color="#f59e0b"]

            txn -> who -> sparse -> pop -> ready
        }
    """)
    st.markdown(
        """
        Before building recommendation models, we need to understand:
        - **How many products does a typical customer buy?** (if most buy only 1-2, collaborative filtering won't work well)
        - **How popular are the products?** (a few bestsellers vs many niche items)
        - **How sparse is the user-item matrix?** (recommendation systems live or die by sparsity)
        - **Are there data quality issues?** (cancelled orders, missing IDs, negative quantities)

        These answers determine which algorithms are appropriate and what preprocessing we need.
        """
    )

st.markdown("---")

raw_df = load_raw_transactions()
clean_df = load_clean_transactions()

# ── Dataset Overview ─────────────────────────────────────────────────────────
st.subheader("Dataset Overview — UCI Online Retail")
st.markdown(
    "This is real transaction data from a **UK-based online gift retailer** (2010–2011). "
    "Each row is one line item from an invoice — a customer buying a specific product."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{len(raw_df):,}")
c2.metric("After Cleaning", f"{len(clean_df):,}")
c3.metric("Unique Customers", f"{clean_df['CustomerID'].nunique():,}")
c4.metric("Unique Products", f"{clean_df['StockCode'].nunique():,}")

st.dataframe(clean_df.head(100), use_container_width=True, height=250)

st.markdown(
    f"**Data cleaning removed {len(raw_df) - len(clean_df):,} rows** — "
    f"cancelled orders (InvoiceNo starting with 'C'), rows with no CustomerID, "
    f"and negative quantities/prices."
)

st.markdown("---")

# ── Purchase Distribution ────────────────────────────────────────────────────
st.subheader("How Many Products Does Each Customer Buy?")

user_counts = clean_df.groupby("CustomerID")["StockCode"].nunique().reset_index()
user_counts.columns = ["CustomerID", "UniqueProducts"]

fig_user = px.histogram(
    user_counts, x="UniqueProducts", nbins=60,
    labels={"UniqueProducts": "Unique Products Purchased"},
    color_discrete_sequence=["#636EFA"],
)
fig_user.update_layout(title="Distribution of Products per Customer", yaxis_title="Number of Customers")
st.plotly_chart(fig_user, use_container_width=True)

median_products = user_counts["UniqueProducts"].median()
mean_products = user_counts["UniqueProducts"].mean()
st.info(
    f"**Median customer buys {median_products:.0f} unique products**, "
    f"mean is {mean_products:.0f}. This is excellent for collaborative filtering — "
    f"we have enough purchase history per customer to find meaningful patterns."
)

st.markdown("---")

# ── Product Popularity ───────────────────────────────────────────────────────
st.subheader("Product Popularity — The Long Tail")

item_counts = clean_df.groupby("Description")["CustomerID"].nunique().reset_index()
item_counts.columns = ["Product", "UniqueBuyers"]
item_counts = item_counts.sort_values("UniqueBuyers", ascending=False).reset_index(drop=True)

fig_tail = go.Figure()
fig_tail.add_trace(go.Bar(
    x=list(range(len(item_counts))),
    y=item_counts["UniqueBuyers"].values,
    marker_color="#636EFA",
))
fig_tail.update_layout(
    title="Product Popularity (sorted by unique buyers)",
    xaxis_title="Product Rank",
    yaxis_title="Unique Buyers",
    yaxis_type="log",
    height=350,
)
st.plotly_chart(fig_tail, use_container_width=True)

st.markdown(
    """
    **This is the "long tail" pattern** — a small number of products are very popular,
    while most products are bought by only a few customers.

    This is typical in e-commerce and has implications:
    - **Popular items** are easy to recommend (everyone buys them) but not very useful
    - **Niche items** are where recommendation systems add the most value — finding
      the right niche product for the right customer
    """
)

st.markdown("**Top 15 Products by Unique Buyers:**")
st.dataframe(item_counts.head(15), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Sparsity Analysis ────────────────────────────────────────────────────────
st.subheader("The Sparsity Problem")

interactions, *_ = build_interaction_matrix()
n_users, n_items = interactions.shape
n_possible = n_users * n_items
n_actual = (interactions > 0).sum().sum()
sparsity = 1 - (n_actual / n_possible)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Users", f"{n_users:,}")
sc2.metric("Items", f"{n_items:,}")
sc3.metric("Possible Pairs", f"{n_possible:,}")
sc4.metric("Actual Interactions", f"{n_actual:,}")

st.metric("Matrix Sparsity", f"{sparsity:.2%}")

fig_sparse = go.Figure()
fig_sparse.add_trace(go.Bar(
    x=["Filled (interactions)", "Empty (unknown)"],
    y=[n_actual, n_possible - n_actual],
    marker_color=["#10b981", "#e5e7eb"],
    text=[f"{n_actual:,}", f"{n_possible - n_actual:,}"],
    textposition="auto",
))
fig_sparse.update_layout(
    title="User-Item Matrix: Filled vs Empty Cells",
    yaxis_title="Number of Cells",
    yaxis_type="log",
    height=350,
)
st.plotly_chart(fig_sparse, use_container_width=True)

st.markdown(
    f"""
    **{sparsity:.2%} of the matrix is empty.** This means out of all possible
    user-item combinations, we only know about a tiny fraction.

    **This is the fundamental challenge of recommendation systems.** We need to
    predict what goes in those empty cells — "would this customer like this product?"

    Matrix factorization solves this by finding hidden patterns (latent factors) that
    explain the observed purchases. If Customer A and Customer B both buy gift sets
    and candles, the model learns they have similar taste — and can recommend items
    one bought that the other hasn't seen yet.
    """
)

st.markdown("---")

# ── Sample of the User-Item Matrix ──────────────────────────────────────────
st.subheader("What the User-Item Matrix Looks Like")
st.markdown(
    "This is the core data structure for recommendation. Each row is a customer, "
    "each column is a product. The numbers are purchase quantities. **Zero means "
    "the customer hasn't bought that product (yet).**"
)

sample_users = interactions.index[:8]
top_items = interactions.sum().nlargest(12).index
sample_matrix = interactions.loc[sample_users, top_items]
sample_matrix.columns = [f"{c[:15]}..." if len(str(c)) > 15 else c for c in sample_matrix.columns]

fig_heatmap = px.imshow(
    sample_matrix.values,
    x=list(sample_matrix.columns),
    y=[str(u) for u in sample_matrix.index],
    color_continuous_scale="YlOrRd",
    text_auto=True,
    aspect="auto",
)
fig_heatmap.update_layout(
    title="User-Item Matrix Sample (8 customers × 12 popular products)",
    xaxis_title="Product (StockCode)",
    yaxis_title="CustomerID",
    height=400,
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown(
    "**Most cells are 0** — that's the sparsity we discussed. "
    "The goal of our recommendation models is to predict meaningful values for those zeros."
)

st.markdown("---")

# ── Revenue and Time Patterns ────────────────────────────────────────────────
st.subheader("Purchase Patterns Over Time")

daily = clean_df.groupby(clean_df["InvoiceDate"].dt.date).agg(
    orders=("InvoiceNo", "nunique"),
    revenue=("TotalPrice", "sum"),
).reset_index()
daily.columns = ["Date", "Orders", "Revenue"]

fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=["Daily Orders", "Daily Revenue"])
fig_time.add_trace(go.Scatter(x=daily["Date"], y=daily["Orders"], mode="lines", line_color="#636EFA", name="Orders"), row=1, col=1)
fig_time.add_trace(go.Scatter(x=daily["Date"], y=daily["Revenue"], mode="lines", line_color="#10b981", name="Revenue"), row=2, col=1)
fig_time.update_layout(height=450, showlegend=False)
fig_time.update_yaxes(title_text="Orders", row=1, col=1)
fig_time.update_yaxes(title_text="Revenue (£)", row=2, col=1)
st.plotly_chart(fig_time, use_container_width=True)

st.markdown(
    "**Business insight:** Clear seasonal spikes suggest gift-buying periods. "
    "A recommendation system should account for temporal trends — what's popular "
    "in November (holiday shopping) differs from July."
)

# ── Country distribution ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Customer Geography")
country_counts = clean_df.groupby("Country")["CustomerID"].nunique().sort_values(ascending=False).reset_index()
country_counts.columns = ["Country", "Unique Customers"]

fig_country = px.bar(
    country_counts.head(15), x="Country", y="Unique Customers",
    color="Unique Customers", color_continuous_scale="Viridis",
)
fig_country.update_layout(title="Top 15 Countries by Unique Customers", coloraxis_showscale=False, height=350)
st.plotly_chart(fig_country, use_container_width=True)

uk_pct = country_counts.iloc[0]["Unique Customers"] / country_counts["Unique Customers"].sum() * 100
st.info(
    f"**{uk_pct:.0f}% of customers are from the UK.** This is important context — "
    f"our recommendation model will primarily reflect UK shopping preferences."
)

# ═══════════════════════════════════════════════════════════════════════════════
# Customer Segmentation (KMeans on purchase behavior)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Customer Segments (KMeans Clustering)")
st.markdown(
    "We use **KMeans** to group customers based on their purchase behavior. "
    "Unlike the churn project (which clusters on demographics), here we cluster on "
    "**how much they buy, how often, and how much they spend** — the classic RFM approach."
)

with st.expander("What is RFM and why cluster on it?", expanded=False):
    st.markdown(
        """
        **RFM** stands for **Recency, Frequency, Monetary** — three metrics that describe
        a customer's purchasing behavior:

        | Metric | What it measures | Our calculation |
        |---|---|---|
        | **Recency** | How recently did they buy? | Days since last purchase |
        | **Frequency** | How often do they buy? | Number of unique orders |
        | **Monetary** | How much do they spend? | Total revenue from this customer |

        **Why these 3?** They capture the most important dimensions of customer value:
        - A customer who bought yesterday (low recency) is more engaged than one from 6 months ago
        - A customer with 50 orders (high frequency) is more loyal than a one-time buyer
        - A customer who spent £5,000 (high monetary) is more valuable than one who spent £20

        **KMeans groups customers who are similar across all 3 dimensions simultaneously.**
        This reveals natural segments like "VIPs" (recent + frequent + high spend),
        "at-risk" (not recent + was frequent), and "one-timers" (one old purchase).
        """
    )

with st.expander("How does KMeans work on this data?", expanded=False):
    st.graphviz_chart("""
        digraph kmeans_rec {
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.25,0.12"]
            edge [color="#888888", penwidth=1.5]

            txn [label="397K transactions\\nper customer", fillcolor="#dbeafe", color="#3b82f6"]
            rfm [label="Compute RFM\\nRecency, Frequency,\\nMonetary per customer", fillcolor="#e0e7ff", color="#6366f1"]
            scale [label="Scale features\\n(StandardScaler)\\nso all 3 are equal", fillcolor="#fce7f3", color="#ec4899"]
            init [label="Place k centroids\\nrandomly in 3D space", fillcolor="#fef3c7", color="#f59e0b"]
            assign [label="Assign each customer\\nto nearest centroid\\n(Euclidean distance)", fillcolor="#d1fae5", color="#10b981"]
            update [label="Move centroids to\\naverage of their\\nassigned customers", fillcolor="#d1fae5", color="#10b981"]
            check [label="Converged?", fillcolor="#fee2e2", color="#ef4444"]
            done [label="Each customer\\nhas a segment label", fillcolor="#d1fae5", color="#10b981"]

            txn -> rfm -> scale -> init -> assign -> update -> check
            check -> assign [label="No", style=dashed]
            check -> done [label="Yes"]
        }
    """)
    st.markdown(
        """
        **Distance calculation:** KMeans measures how "close" two customers are using
        Euclidean distance across all 3 scaled features:
        """
    )
    st.latex(r"d = \sqrt{(\text{recency}_1 - \text{recency}_2)^2 + (\text{frequency}_1 - \text{frequency}_2)^2 + (\text{monetary}_1 - \text{monetary}_2)^2}")
    st.markdown(
        """
        Customers with similar recency, frequency, AND monetary values end up in the same cluster.
        The algorithm iterates (assign → update → check) until centroids stabilize, usually in 5–20 rounds.
        """
    )

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as StdScaler

@st.cache_data
def compute_rfm_segments(transactions: pd.DataFrame, n_clusters: int = 4):
    ref_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = transactions.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    scaler = StdScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    return rfm, km, scaler, rfm_scaled

n_rec_clusters = st.slider("Number of customer segments", 2, 8, 4, key="rec_clusters")
rfm_df, km_model, rfm_scaler, rfm_scaled = compute_rfm_segments(clean_df, n_rec_clusters)

# ── Scatter plot (Frequency vs Monetary, color by cluster) ───────────────────
fig_rfm = px.scatter(
    rfm_df, x="Frequency", y="Monetary", color=rfm_df["Cluster"].astype(str),
    size="Monetary", size_max=15, opacity=0.6,
    hover_data=["CustomerID", "Recency"],
    labels={"color": "Cluster"},
)
fig_rfm.update_layout(
    title="Customer Segments — Frequency vs Monetary Value (color = cluster)",
    xaxis_title="Frequency (number of orders)",
    yaxis_title="Monetary (total spend £)",
    height=450,
)
st.plotly_chart(fig_rfm, use_container_width=True)

# ── Cluster summary table ────────────────────────────────────────────────────
st.markdown("### What Each Segment Represents")

rec_cluster_stats = []
rec_profiles = []
for c in range(n_rec_clusters):
    cdf = rfm_df[rfm_df["Cluster"] == c]
    avg_r = cdf["Recency"].mean()
    avg_f = cdf["Frequency"].mean()
    avg_m = cdf["Monetary"].mean()

    if avg_f > rfm_df["Frequency"].median() * 2 and avg_m > rfm_df["Monetary"].median() * 2:
        label = "VIP / Champions"
    elif avg_r > rfm_df["Recency"].median() * 1.5 and avg_f > rfm_df["Frequency"].median():
        label = "At-Risk (were active)"
    elif avg_f <= 2 and avg_m < rfm_df["Monetary"].median():
        label = "One-Timers"
    elif avg_r < rfm_df["Recency"].median() and avg_f > rfm_df["Frequency"].median():
        label = "Loyal Regulars"
    elif avg_m > rfm_df["Monetary"].median():
        label = "Big Spenders"
    else:
        label = "Casual Shoppers"

    rec_cluster_stats.append({
        "Segment": f"{c} — {label}",
        "Customers": f"{len(cdf):,}",
        "Avg Recency": f"{avg_r:.0f} days",
        "Avg Frequency": f"{avg_f:.1f} orders",
        "Avg Monetary": f"£{avg_m:,.0f}",
    })
    rec_profiles.append({"cluster": c, "label": label, "size": len(cdf),
                         "recency": avg_r, "frequency": avg_f, "monetary": avg_m})

st.dataframe(pd.DataFrame(rec_cluster_stats), use_container_width=True, hide_index=True)

best_seg = max(rec_profiles, key=lambda x: x["monetary"])
risk_seg = max(rec_profiles, key=lambda x: x["recency"])

st.markdown(
    f"""
    **Interpreting the segments:**

    - **Most valuable segment:** Cluster {best_seg['cluster']} ({best_seg['label']}) —
      {best_seg['size']:,} customers spending avg £{best_seg['monetary']:,.0f} with
      {best_seg['frequency']:.1f} orders. **These customers drive revenue — keep them happy
      with personalized recommendations.**

    - **Highest risk segment:** Cluster {risk_seg['cluster']} ({risk_seg['label']}) —
      {risk_seg['size']:,} customers with avg {risk_seg['recency']:.0f} days since last purchase.
      **These customers are going dormant — win them back with targeted product recommendations
      based on their past purchases.**
    """
)

with st.expander("How do segments help the recommendation system?", expanded=False):
    st.markdown(
        """
        **Different segments need different recommendation strategies:**

        | Segment | Recommendation Strategy |
        |---|---|
        | **VIP / Champions** | Cross-sell premium items, early access to new products, personalized bundles |
        | **Loyal Regulars** | Recommend complementary products to what they already buy |
        | **One-Timers** | Recommend popular items similar to their single purchase (cold-start) |
        | **At-Risk** | Re-engagement emails with their previously purchased categories |
        | **Big Spenders** | High-value product recommendations, exclusive collections |
        | **Casual Shoppers** | Best-sellers and trending items to increase engagement |

        **The combination is powerful:** Matrix factorization (SVD, ALS) finds *what* to
        recommend. Clustering tells us *how* to present it and *when* to reach out.
        A VIP gets a different email template and product mix than a one-timer, even if
        the underlying algorithm is the same.
        """
    )

# ── Centroid radar chart ─────────────────────────────────────────────────────
st.markdown("### Centroid Profiles")
st.markdown(
    "Each centroid represents the \"average customer\" in that segment. "
    "The radar chart shows how segments differ across all 3 RFM dimensions."
)

import plotly.graph_objects as go_radar
rfm_features = ["Recency", "Frequency", "Monetary"]
categories_r = rfm_features + [rfm_features[0]]
fig_rfm_radar = go_radar.Figure()
for i in range(n_rec_clusters):
    vals = km_model.cluster_centers_[i].tolist()
    vals.append(vals[0])
    fig_rfm_radar.add_trace(go_radar.Scatterpolar(
        r=vals, theta=categories_r, fill="toself",
        name=f"Cluster {i} — {rec_profiles[i]['label']}",
    ))
fig_rfm_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Segment Centroids (scaled: 0 = average, positive = above, negative = below)",
    height=400,
)
st.plotly_chart(fig_rfm_radar, use_container_width=True)

st.markdown(
    "**Reading the radar:** A segment that extends far on Frequency and Monetary but stays "
    "near zero on Recency contains recent, frequent, high-spending customers (your best segment). "
    "A segment far on Recency but low on Frequency/Monetary represents dormant one-time buyers."
)
